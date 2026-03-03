import redis
import os
import boto3
from botocore.client import Config as BotoConfig
import cv2
import json
import numpy as np
import mediapipe as mp

# --- Configuration ---
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')

S3_CONFIG = {
    "endpoint_url": os.getenv('S3_ENDPOINT', "http://minio:9000"),
    "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
    "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
    "config": BotoConfig(signature_version='s3v4'),
    "region_name": 'eu-west-1'
}

tolerance_max=float(os.getenv('RATIO',0.60))

# Seuils de détection Head-Pose
X_THRESHOLD = 0.4   # Droite > 0.4 / Gauche < -0.4
Y_DOWN = 0.30       # Bas > 0.30
Y_UP = -0.1         # Haut < -0.1

def analyze_head_pose(filepath):
    """Analyse la vidéo et retourne la séquence de mouvements détectée."""
    telemetry = []
    movement_history = []
    last_state = None
    
    # Indices MediaPipe
    NOSE = 4
    L_EYE = 33
    R_EYE = 263

    cap = cv2.VideoCapture(filepath)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(refine_landmarks=True, static_image_mode=False) as face_mesh:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            # Conversion BGR vers RGB pour MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0]
                p_nose = mesh.landmark[NOSE]
                p_l = mesh.landmark[L_EYE]
                p_r = mesh.landmark[R_EYE]

                # Calcul du centre des yeux et de l'échelle
                mid_x = (p_l.x + p_r.x) / 2
                mid_y = (p_l.y + p_r.y) / 2
                dist = np.sqrt((p_r.x - p_l.x)**2 + (p_r.y - p_l.y)**2)

                # Calcul des vecteurs relatifs (Head Pose)
                rel_x = (p_nose.x - mid_x) / dist
                rel_y = (p_nose.y - mid_y) / dist

                # Détermination de l'état (Logic de mouvement)
                state = "CENTRE"
                if rel_x > X_THRESHOLD: state = "DROITE"
                elif rel_x < -X_THRESHOLD: state = "GAUCHE"
                elif rel_y > Y_DOWN: state = "BAS"
                elif rel_y < Y_UP: state = "HAUT"

                # Enregistrement si changement d'état
                if state != last_state:
                    movement_history.append({
                        "frame": frame_idx,
                        "direction": state,
                        "raw": {"x": round(rel_x, 3), "y": round(rel_y, 3)}
                    })
                    last_state = state

    cap.release()
    return movement_history

def start_worker():
    r = redis.from_url(REDIS_URL, decode_responses=True)
    s3 = boto3.client('s3', **S3_CONFIG)
    
    pubsub = r.pubsub()
    pubsub.subscribe('ia_jobs')
    print("🚀 Worker IA prêt - En attente de messages sur 'ia_jobs'...")

    for msg in pubsub.listen():
        if msg['type'] != 'message': continue
        
        try:
            job = json.loads(msg['data'])
            challenge =job['challenge'].split(",")
            print(f"challenge : {challenge}")

            filename = job['filename']
            local_path = f"/tmp/{filename}"
            
            print(f"📥 Traitement de {filename} pour l'utilisateur {job.get('user_id')}")
            
            # 1. Download
            s3.download_file("pad-bucket", filename, local_path)
            
            # 2. Process
            movements = analyze_head_pose(local_path)

            directions_only = [m['direction'] for m in movements if m['direction']!='CENTRE']
            cpt_good = 0
            for i in range(len(directions_only)) : 
                if directions_only[i] == challenge[i]:
                    cpt_good+=1
            ratio = cpt_good/len(challenge)

            print(f"ratio de good : {ratio}, ratio prédéfini : {tolerance_max} : ratiotype : {ratio} {tolerance_max}")
            if ratio < tolerance_max :

            # 3. Publication du résultat
                response = {
                    "user_id": job.get('user_id'),
                    "status": "IA_FAIL",
                    "filename": filename,
                    "movements": movements
                }
            else :
                response = {
                    "user_id": job.get('user_id'),
                    "status": "IA_SUCCESS",
                    "filename": filename,
                    "movements": movements
                }
            r.publish('ia_results', json.dumps(response))

            print(f"Job terminé : {len(movements)} changements de pose détectés. {directions_only}")
            for i in directions_only : 
                if i == challenge : 
                    print(f"i == challenge c'est crazy : {challenge} ;{i}")
        except Exception as e:
            error_msg = {"status": "IA_ERROR", "error": str(e), "filename": filename if 'filename' in locals() else "unknown"}
            r.publish('ia_results', json.dumps(error_msg))
            print(f"Erreur : {e}")
            
        finally: 
            if 'local_path' in locals() and os.path.exists(local_path):
                os.remove(local_path)

if __name__ == "__main__":
    start_worker()