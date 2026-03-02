import cv2
import mediapipe as mp
import numpy as np
import redis, json, os, boto3
from botocore.client import Config as BotoConfig

# ==================== CONFIGURATION DE PRÉCISION ====================
class Config:
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    
    # SEUILS (En unités de "distance entre les yeux")
    # On augmente pour ne prendre que les mouvements TRÈS nets
    H_THRESHOLD = 0.45  
    V_THRESHOLD = 0.30  
    
    # ZONE NEUTRE (Large pour ne jamais rester bloqué)
    NEUTRAL_LIMIT = 0.15 
    
    # Sécurité : Si le nez bouge de plus de 2 visages, c'est du bruit
    SANITY_CHECK = 2.0 

class UnbreakableAnalyzer:
    def __init__(self, path, challenge):
        self.path = path
        self.challenge = [c for c in challenge.split(',') if c]
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.neutral_x, self.neutral_y = None, None
        self.calib_buf = []
        self.seq = []
        self.state = "NEUTRAL"

    def get_coords(self, landmarks):
        """ Calcule la position relative du nez normalisée """
        lm = landmarks.landmark
        # Points : Nez(4), OeilG(33), OeilD(263)
        nose = np.array([lm[4].x, lm[4].y])
        eye_l = np.array([lm[33].x, lm[33].y])
        eye_r = np.array([lm[263].x, lm[263].y])
        
        # Unité de mesure : distance entre les yeux
        dist_eyes = np.linalg.norm(eye_l - eye_r)
        if dist_eyes < 0.01: return None, None # Visage trop loin ou perdu
        
        center_eyes = (eye_l + eye_r) / 2
        # Calcul de l'offset relatif (Indépendant de la résolution)
        curr_x = (nose[0] - center_eyes[0]) / dist_eyes
        curr_y = (nose[1] - center_eyes[1]) / dist_eyes
        
        return curr_x, curr_y

    def analyze(self):
        cap = cv2.VideoCapture(self.path)
        f_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            f_idx += 1

            res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks: continue

            cx, cy = self.get_coords(res.multi_face_landmarks[0])
            if cx is None or abs(cx) > Config.SANITY_CHECK: continue

            # --- 1. CALIBRATION DYNAMIQUE ---
            if self.neutral_x is None:
                self.calib_buf.append((cx, cy))
                if len(self.calib_buf) >= 15:
                    self.neutral_x = sum(i[0] for i in self.calib_buf) / 15
                    self.neutral_y = sum(i[1] for i in self.calib_buf) / 15
                    print(f"🎯 Calibré ! Neutre: X={self.neutral_x:.2f}")
                continue

            # --- 2. CALCUL DES ÉCARTS ---
            dx = cx - self.neutral_x
            dy = cy - self.neutral_y

            # --- 3. MACHINE À ÉTATS (Hystérésis) ---
            if self.state == "NEUTRAL":
                move = None
                if dx > Config.H_THRESHOLD: move = "DROITE"
                elif dx < -Config.H_THRESHOLD: move = "GAUCHE"
                elif dy > Config.V_THRESHOLD: move = "BAS"
                elif dy < -Config.V_THRESHOLD: move = "HAUT"
                
                if move:
                    self.seq.append(move)
                    self.state = move
                    print(f"📍 VALIDÉ : {move} (dx:{dx:.2f})")
            else:
                # On attend le retour au centre pour débloquer
                if abs(dx) < Config.NEUTRAL_LIMIT and abs(dy) < Config.NEUTRAL_LIMIT:
                    self.state = "NEUTRAL"
                    print(f"🏠 Centre OK (frame {f_idx})")

        cap.release()
        it = iter(self.seq)
        success = all(d in it for d in self.challenge)
        return success, self.seq

# ==================== RUNNER ====================
r = redis.Redis(host=Config.REDIS_HOST, port=6379, decode_responses=True)
s3 = boto3.client('s3', endpoint_url="http://minio:9000", aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin", config=BotoConfig(signature_version='s3v4'), region_name='eu-west-1')

def start():
    pubsub = r.pubsub()
    pubsub.subscribe('ia_jobs')
    print("🚀 Worker IA 'Unbreakable' Ready")

    for msg in pubsub.listen():
        if msg['type'] != 'message': continue
        job = json.loads(msg['data'])
        filename = job['filename']
        path = f"/tmp/{filename}"
        
        try:
            s3.download_file("pad-bucket", filename, path)
            ok, seq = UnbreakableAnalyzer(path, job.get('challenge', '')).analyze()
            
            res = {"user_id": job['user_id'], "status": "IA_SUCCESS" if ok else "IA_ERROR", "filename": filename}
            r.publish('ia_results', json.dumps(res))
            print(f"📤 Final: {ok} | Seq: {seq}")
        except Exception as e: print(f"❌ Error: {e}")
        finally: 
            if os.path.exists(path): os.remove(path)

if __name__ == "__main__": start()