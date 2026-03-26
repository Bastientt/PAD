import redis
import os
import boto3
from deepfake_detector import run_deepfake_analysis
from botocore.client import Config as BotoConfig
import cv2
import json
import numpy as np
import mediapipe as mp
from collections import deque

# --- Configuration ---
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')

S3_CONFIG = {
    "endpoint_url": os.getenv('S3_ENDPOINT', "http://minio:9000"),
    "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
    "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
    "config": BotoConfig(signature_version='s3v4'),
    "region_name": 'eu-west-1'
}

tolerance_max = float(os.getenv('RATIO', 0.60))

# ════════════════════════════════════════════════════════════════
# PARAMÈTRES — commentés pour faciliter le tuning
# ════════════════════════════════════════════════════════════════

# ── Seuils directionnels ─────────────────────────────────────────
X_THRESHOLD = 0.45
Y_DOWN      = 0.32
Y_UP        = -0.08   # plus sensible : quand on lève la tête, MediaPipe perd vite le visage

# Zone morte de retour au centre (hystérésis)
X_CENTER_ZONE  = 0.28
Y_CENTER_DOWN  = 0.18
Y_CENTER_UP    = -0.02  # asymétrique : on lâche HAUT plus facilement

# ── Lissage et debounce ───────────────────────────────────────────
SMOOTHING_FRAMES = 4   # moyenne glissante sur N frames
CONFIRM_FRAMES   = 5   # frames consécutives pour confirmer un état

# ── Robustesse face-tracking ──────────────────────────────────────
# Frames sans visage tolérées avant de reset le candidat
FACE_GRACE_FRAMES = 20  # élevé car tête levée = perte fréquente du visage

# ── Preprocessing image ───────────────────────────────────────────
CLAHE_CLIP    = 2.0
CLAHE_TILE    = (8, 8)
RESIZE_WIDTH  = 640   # None pour désactiver


# ════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════

def build_clahe():
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

def preprocess_frame(frame, clahe, target_width=RESIZE_WIDTH):
    """
    Pipeline de preprocessing pour éclairage difficile / cam téléphone :
    1. Resize   → résolution homogène
    2. CLAHE    → rehausse le contraste localement (espace LAB)
    3. Bilatéral → débruitage sans flouter les contours du visage
    """
    if target_width is not None:
        h, w = frame.shape[:2]
        if w != target_width:
            ratio = target_width / w
            frame = cv2.resize(frame, (target_width, int(h * ratio)), interpolation=cv2.INTER_LINEAR)

    lab       = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    lab_eq    = cv2.merge([clahe.apply(l), a, b])
    frame_eq  = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    frame_dn  = cv2.bilateralFilter(frame_eq, d=5, sigmaColor=35, sigmaSpace=35)

    return cv2.cvtColor(frame_dn, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════════════════════
# LOGIQUE DE DÉTECTION
# ════════════════════════════════════════════════════════════════

def _smooth(buffer):
    arr = np.array(buffer)
    return arr[:, 0].mean(), arr[:, 1].mean()

def _get_state(rel_x, rel_y, current_state):
    if current_state == "CENTRE":
        if   rel_x >  X_THRESHOLD: return "DROITE"
        elif rel_x < -X_THRESHOLD: return "GAUCHE"
        elif rel_y >  Y_DOWN:      return "BAS"
        elif rel_y <  Y_UP:        return "HAUT"
        return "CENTRE"
    else:
        in_center = (
            abs(rel_x) < X_CENTER_ZONE
            and rel_y  < Y_CENTER_DOWN
            and rel_y  > Y_CENTER_UP
        )
        if in_center: return "CENTRE"
        if   rel_x >  X_THRESHOLD: return "DROITE"
        elif rel_x < -X_THRESHOLD: return "GAUCHE"
        elif rel_y >  Y_DOWN:      return "BAS"
        elif rel_y <  Y_UP:        return "HAUT"
        return current_state


# ════════════════════════════════════════════════════════════════
# ANALYSE PRINCIPALE
# ════════════════════════════════════════════════════════════════

NOSE  = 4
L_EYE = 33
R_EYE = 263

def analyze_head_pose(filepath):
    print(f"🔍 [IA] Analyse : {filepath}")
    movement_history     = []
    last_confirmed_state = "CENTRE"
    candidate_state      = "CENTRE"
    candidate_count      = 0
    face_lost_frames     = 0
    coord_buffer         = deque(maxlen=SMOOTHING_FRAMES)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la vidéo")
        return []

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    clahe     = build_clahe()

    print(f"   📹 {total} frames @ {fps:.1f} fps")

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            rgb     = preprocess_frame(frame, clahe)
            results = face_mesh.process(rgb)

            # ── Gestion de la perte de visage ────────────────────────
            if not results.multi_face_landmarks:
                face_lost_frames += 1
                if face_lost_frames > FACE_GRACE_FRAMES:
                    candidate_count = 0  # reset candidat mais pas l'état confirmé
                if frame_idx % 30 == 0:
                    print(f"   ⚠️  Frame {frame_idx}: pas de visage ({face_lost_frames} consécutives)")
                continue

            face_lost_frames = 0

            mesh = results.multi_face_landmarks[0]
            p_n  = mesh.landmark[NOSE]
            p_l  = mesh.landmark[L_EYE]
            p_r  = mesh.landmark[R_EYE]

            mid_x = (p_l.x + p_r.x) / 2
            mid_y = (p_l.y + p_r.y) / 2
            dist  = np.sqrt((p_r.x - p_l.x)**2 + (p_r.y - p_l.y)**2)
            if dist < 1e-6:
                continue

            # Caméra frontale = image miroir → on inverse X
            raw_x = -((p_n.x - mid_x) / dist)
            raw_y = (p_n.y - mid_y) / dist

            coord_buffer.append((raw_x, raw_y))
            if len(coord_buffer) < SMOOTHING_FRAMES:
                continue
            rel_x, rel_y = _smooth(coord_buffer)

            raw_state = _get_state(rel_x, rel_y, last_confirmed_state)

            if raw_state == candidate_state:
                candidate_count += 1
            else:
                candidate_state = raw_state
                candidate_count = 1

            if candidate_count == CONFIRM_FRAMES:
                if candidate_state != last_confirmed_state:
                    print(
                        f"   🎥 Frame {frame_idx}: "
                        f"{last_confirmed_state} → {candidate_state} "
                        f"(x={rel_x:+.3f}, y={rel_y:+.3f})"
                    )
                    movement_history.append({
                        "frame":     frame_idx,
                        "direction": candidate_state,
                        "raw":       {"x": round(rel_x, 3), "y": round(rel_y, 3)},
                    })
                    last_confirmed_state = candidate_state

    cap.release()
    print(f"✅ Terminé — {len(movement_history)} mouvements confirmés")
    return movement_history


# ════════════════════════════════════════════════════════════════
# WORKER
# ════════════════════════════════════════════════════════════════

def start_worker():
    r  = redis.from_url(REDIS_URL, decode_responses=True)
    s3 = boto3.client('s3', **S3_CONFIG)

    pubsub = r.pubsub()
    pubsub.subscribe('ia_jobs')
    print("🚀 [WORKER] Prêt — écoute 'ia_jobs'...")

    for msg in pubsub.listen():
        if msg['type'] != 'message':
            continue

        current_user_id = "unknown"
        filename        = "unknown"
        local_path      = None

        try:
            print(f"\n📩 [REDIS] {msg['data']}")
            job = json.loads(msg['data'])

            current_user_id = job.get('user_id', "unknown")
            filename        = job.get('filename', "unknown")
            challenge       = job['challenge'].split(",")

            print(f"📝 User: {current_user_id} | Challenge: {challenge}")

            local_path = f"/tmp/{filename}"

            print(f"⬇️  Téléchargement {filename}…")
            s3.download_file("pad-bucket", filename, local_path)
            print("   ✅ OK")

            # ── Vérification anti-deepfake ──────────────────────────
            deepfake_result = run_deepfake_analysis(filename)
            r.publish('ia_results', json.dumps({
                "user_id":  current_user_id,
                "status":   "DEEPFAKE_OK",
                "filename": filename,
                "deepfake": deepfake_result,
            }))

            movements       = analyze_head_pose(local_path)
            directions_only = [m['direction'] for m in movements if m['direction'] != 'CENTRE']

            print(f"📊 Séquence détectée : {directions_only}")
            print(f"📋 Challenge attendu : {challenge}")

            len_challenge = len(challenge)
            len_detected  = len(directions_only)
            limit         = min(len_challenge, len_detected)
            cpt_good      = 0

            print(f"⚖️  Comparaison {limit}/{len_challenge} étapes")
            for i in range(limit):
                match = "✅" if directions_only[i] == challenge[i] else "❌"
                print(f"   {match} Step {i}: attendu={challenge[i]} obtenu={directions_only[i]}")
                if directions_only[i] == challenge[i]:
                    cpt_good += 1

            ratio  = cpt_good / len_challenge if len_challenge > 0 else 0
            status = "IA_SUCCESS" if ratio >= tolerance_max else "IA_FAIL"

            print(f"📈 Score: {cpt_good}/{len_challenge} ({ratio:.0%}) — seuil {tolerance_max:.0%} → {status}")

            r.publish('ia_results', json.dumps({
                "user_id":   current_user_id,
                "status":    status,
                "filename":  filename,
                "movements": movements,
            }))

        except Exception as e:
            import traceback
            print(f"❌ CRASH: {e}")
            traceback.print_exc()
            r.publish('ia_results', json.dumps({
                "user_id":  current_user_id,
                "status":   "IA_ERROR",
                "error":    str(e),
                "filename": filename,
            }))

        finally:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
                print(f"🧹 Cleanup: {local_path}")


if __name__ == "__main__":
    start_worker()
