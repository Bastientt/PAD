import cv2
import mediapipe as mp
import numpy as np
import redis, json, os, boto3, sys
from botocore.client import Config as BotoConfig
from collections import deque

# ==================== CONFIGURATION DE PRÉCISION ====================
class Config:
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    
# --- PARAMETRES DE PRECISION BOOSTES ---
    H_THRESHOLD = 0.25   # Sensibilité horizontale (Optimisée)
    V_THRESHOLD = 0.20   # Sensibilité verticale (Optimisée)
    NEUTRAL_LIMIT = 0.12 # Zone de retour au centre
    
    # --- STABILITE (MOYENNE GLISSANTE) ---
    SMOOTHING_WINDOW = 5 # Lissage pour éviter les micro-sauts
    CALIBRATION_FRAMES = 10 # Rapidité de démarrage
    
    # --- SECURITE ---
    SANITY_CHECK = 1.5

class UnbreakableAnalyzer:
    def __init__(self, path, challenge):
        # Utilisation de os.path.normpath pour la compatibilité Mac/Linux/Windows
        self.path = os.path.normpath(path)
        self.challenge = [c.upper().strip() for c in challenge.split(',') if c]
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        
        self.neutral_x, self.neutral_y = None, None
        self.calib_buf = []
        self.seq = []
        self.state = "NEUTRAL"
        
        # Files pour le lissage des coordonnées
        self.history_x = deque(maxlen=Config.SMOOTHING_WINDOW)
        self.history_y = deque(maxlen=Config.SMOOTHING_WINDOW)

    def get_coords(self, landmarks):
        lm = landmarks.landmark
        nose = np.array([lm[4].x, lm[4].y])
        eye_l = np.array([lm[33].x, lm[33].y])
        eye_r = np.array([lm[263].x, lm[263].y])
        
        dist_eyes = np.linalg.norm(eye_l - eye_r)
        if dist_eyes < 0.01: return None, None 
        
        center_eyes = (eye_l + eye_r) / 2
        curr_x = (nose[0] - center_eyes[0]) / dist_eyes
        curr_y = (nose[1] - center_eyes[1]) / dist_eyes
        
        # Ajout du lissage (Smoothing)
        self.history_x.append(curr_x)
        self.history_y.append(curr_y)
        
        return np.mean(self.history_x), np.mean(self.history_y)

    def analyze(self):
        # CAP_ANY laisse OpenCV choisir le meilleur driver selon l'OS (Mac, Win ou Linux)
        cap = cv2.VideoCapture(self.path, cv2.CAP_ANY)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks: continue

            coords = self.get_coords(res.multi_face_landmarks[0])
            if coords[0] is None or abs(coords[0]) > Config.SANITY_CHECK: continue
            cx, cy = coords

            # 1. CALIBRATION
            if self.neutral_x is None:
                self.calib_buf.append((cx, cy))
                if len(self.calib_buf) >= Config.CALIBRATION_FRAMES:
                    self.neutral_x = np.mean([i[0] for i in self.calib_buf])
                    self.neutral_y = np.mean([i[1] for i in self.calib_buf])
                continue

            # 2. DETECTION
            dx = cx - self.neutral_x
            dy = cy - self.neutral_y

            if self.state == "NEUTRAL":
                move = None
                if dx > Config.H_THRESHOLD: move = "DROITE"
                elif dx < -Config.H_THRESHOLD: move = "GAUCHE"
                elif dy > Config.V_THRESHOLD: move = "BAS"
                elif dy < -Config.V_THRESHOLD: move = "HAUT"
                
                if move:
                    self.seq.append(move)
                    self.state = move
            else:
                if abs(dx) < Config.NEUTRAL_LIMIT and abs(dy) < Config.NEUTRAL_LIMIT:
                    self.state = "NEUTRAL"

        cap.release()
        it = iter(self.seq)
        return all(d in it for d in self.challenge), self.seq

# ==================== RUNNER ====================
def start():
    # Gestion de Redis
    r = redis.Redis(host=Config.REDIS_HOST, port=6379, decode_responses=True)
    
    # Configuration S3 / Minio
    s3 = boto3.client('s3', 
        endpoint_url="http://minio:9000", 
        aws_access_key_id="minioadmin", 
        aws_secret_access_key="minioadmin", 
        config=BotoConfig(signature_version='s3v4'), 
        region_name='eu-west-1'
    )

    pubsub = r.pubsub()
    pubsub.subscribe('ia_jobs')
    print(f"Worker démarré sur {sys.platform}. En attente de jobs...")

    for msg in pubsub.listen():
        if msg['type'] != 'message': continue
        job = json.loads(msg['data'])
        
        # Utilisation de tempfile pour être propre sur Windows et Linux
        import tempfile
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, job['filename'])
        
        try:
            s3.download_file("pad-bucket", job['filename'], path)
            ok, seq = UnbreakableAnalyzer(path, job.get('challenge', '')).analyze()
            
            res = {
                "user_id": job['user_id'], 
                "status": "IA_SUCCESS" if ok else "IA_ERROR", 
                "detected_seq": seq
            }
            r.publish('ia_results', json.dumps(res))
        except Exception as e:
            print(f"Erreur : {e}")
        finally: 
            if os.path.exists(path): os.remove(path)

if __name__ == "__main__":
    start()