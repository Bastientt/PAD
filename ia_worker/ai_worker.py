import cv2
import mediapipe as mp
import numpy as np
import redis, json, os, boto3, time
from botocore.client import Config as BotoConfig

# ==================== CONFIG SIMPLE ====================
class Config:
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    MINIO_URL = "http://minio:9000"
    BUCKET = "pad-bucket"
    
    # SEUILS ULTRA PERMISSIFS (on détecte TOUT d'abord)
    YAW_MIN = 8.0       # Gauche/Droite - TRÈS bas
    PITCH_MIN = 6.0     # Haut/Bas - TRÈS bas
    CENTER_ZONE = 4.0   # Retour au centre
    
    EAR_THRESHOLD = 0.23
    CALIBRATION_FRAMES = 10  # Rapide !

# ==================== DÉTECTEUR SIMPLE ====================
class SimpleLiveness:
    def __init__(self, video_path, challenge):
        self.video_path = video_path
        self.challenge = [c.strip().upper() for c in challenge.split(',') if c.strip()]
        
        # MediaPipe simple
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Plus permissif
            min_tracking_confidence=0.3
        )
        
        # État
        self.neutral_yaw = None
        self.neutral_pitch = None
        self.calib_buffer = []
        
        self.detected = []
        self.current_state = "CENTER"
        self.blink_count = 0
        self.last_ear = 1.0
    
    def get_ear(self, lm):
        """EAR simple"""
        def dist(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        # Œil gauche
        le = [(lm[i].x, lm[i].y) for i in [33, 160, 158, 133, 153, 144]]
        ear_l = (dist(le[1], le[5]) + dist(le[2], le[4])) / (2.0 * dist(le[0], le[3]))
        
        # Œil droit  
        re = [(lm[i].x, lm[i].y) for i in [362, 385, 387, 263, 373, 380]]
        ear_r = (dist(re[1], re[5]) + dist(re[2], re[4])) / (2.0 * dist(re[0], re[3]))
        
        return (ear_l + ear_r) / 2.0
    
    def get_pose_simple(self, lm, w, h):
        """Méthode ULTRA simple - juste les landmarks directs"""
        # Points clés
        nose = lm[1]  # Nez
        left_eye = lm[33]  # Œil gauche
        right_eye = lm[263]  # Œil droit
        chin = lm[152]  # Menton
        
        # YAW (gauche/droite) : asymétrie des yeux
        eye_center_x = (left_eye.x + right_eye.x) / 2
        yaw_raw = (nose.x - eye_center_x) * 100  # Facteur d'échelle
        
        # PITCH (haut/bas) : position relative nez/yeux vs menton
        eye_center_y = (left_eye.y + right_eye.y) / 2
        pitch_raw = (nose.y - eye_center_y) * 100
        
        return yaw_raw, pitch_raw
    
    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la vidéo")
            return False, [], False
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"📹 Vidéo: {total_frames} frames")
        print(f"🎯 Challenge: {' → '.join(self.challenge)}")
        print(f"{'='*60}\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Détection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                if frame_count % 20 == 0:
                    print(f"⚠️  Frame {frame_count}: Pas de visage détecté")
                continue
            
            lm = results.multi_face_landmarks[0].landmark
            
            # Pose simplifiée
            yaw, pitch = self.get_pose_simple(lm, w, h)
            
            # === CALIBRATION ===
            if self.neutral_yaw is None:
                self.calib_buffer.append((yaw, pitch))
                
                if len(self.calib_buffer) >= Config.CALIBRATION_FRAMES:
                    yaws = [y for y, p in self.calib_buffer]
                    pitchs = [p for y, p in self.calib_buffer]
                    
                    self.neutral_yaw = np.median(yaws)
                    self.neutral_pitch = np.median(pitchs)
                    
                    print(f"✅ CALIBRÉ: Y0={self.neutral_yaw:.1f}, P0={self.neutral_pitch:.1f}")
                    print(f"   Range: Y=[{min(yaws):.1f}, {max(yaws):.1f}], P=[{min(pitchs):.1f}, {max(pitchs):.1f}]")
                continue
            
            # Angles relatifs
            rel_yaw = yaw - self.neutral_yaw
            rel_pitch = pitch - self.neutral_pitch
            
            # === CLIGNEMENT ===
            ear = self.get_ear(lm)
            if ear < Config.EAR_THRESHOLD and self.last_ear >= Config.EAR_THRESHOLD:
                self.blink_count += 1
                if "CLIGNE" not in self.detected:
                    self.detected.append("CLIGNE")
                    print(f"👁️  CLIGNE détecté ! (frame {frame_count})")
            self.last_ear = ear
            
            # === DIRECTION ===
            direction = None
            
            # Check centre d'abord
            if abs(rel_yaw) < Config.CENTER_ZONE and abs(rel_pitch) < Config.CENTER_ZONE:
                if self.current_state != "CENTER":
                    print(f"🔵 Retour au centre (frame {frame_count})")
                self.current_state = "CENTER"
            
            # Sinon, check directions
            else:
                # Priorité au mouvement le plus marqué
                if abs(rel_yaw) > abs(rel_pitch):
                    # Mouvement horizontal
                    if rel_yaw < -Config.YAW_MIN:
                        direction = "GAUCHE"
                    elif rel_yaw > Config.YAW_MIN:
                        direction = "DROITE"
                else:
                    # Mouvement vertical
                    if rel_pitch < -Config.PITCH_MIN:
                        direction = "HAUT"
                    elif rel_pitch > Config.PITCH_MIN:
                        direction = "BAS"
                
                # Détection uniquement si changement d'état depuis CENTER
                if direction and self.current_state == "CENTER":
                    if direction not in self.detected:
                        self.detected.append(direction)
                        print(f"✅ {direction} détecté ! (Yaw={rel_yaw:+.1f}, Pitch={rel_pitch:+.1f}) [frame {frame_count}]")
                    self.current_state = direction
            
            # Debug périodique
            if frame_count % 10 == 0:
                print(f"[{frame_count:03d}] Yaw={rel_yaw:+6.1f} Pitch={rel_pitch:+6.1f} | État={self.current_state:8s} | Détectés={self.detected}")
            
            # Check complétion
            if all(move in self.detected for move in self.challenge):
                print(f"\n🎉 CHALLENGE COMPLÉTÉ à la frame {frame_count}/{total_frames} !")
                break
        
        cap.release()
        
        # Résultat
        success = all(move in self.detected for move in self.challenge)
        
        print(f"\n{'='*60}")
        print(f"📊 RÉSULTAT FINAL")
        print(f"{'='*60}")
        print(f"Challenge demandé: {self.challenge}")
        print(f"Mouvements détectés: {self.detected}")
        print(f"Statut: {'✅ SUCCÈS' if success else '❌ ÉCHEC'}")
        print(f"{'='*60}\n")
        
        return success, self.detected, self.blink_count > 0

# ==================== WORKER ====================
def start_worker():
    config = Config()
    r = redis.Redis(host=config.REDIS_HOST, port=6379, decode_responses=True)
    s3 = boto3.client(
        's3',
        endpoint_url=config.MINIO_URL,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=BotoConfig(signature_version='s3v4'),
        region_name='eu-west-1'
    )
    
    pubsub = r.pubsub()
    pubsub.subscribe('ia_jobs')
    
    print("\n" + "="*60)
    print("🚀 SimpleLiveness Worker PRÊT")
    print("="*60 + "\n")
    
    for msg in pubsub.listen():
        if msg['type'] != 'message':
            continue
        
        try:
            job = json.loads(msg['data'])
            user_id = job.get('user_id', 'unknown')
            filename = job['filename']
            challenge = job.get('challenge', 'GAUCHE,DROITE,HAUT,BAS')
            
            print(f"\n📥 JOB REÇU")
            print(f"   User: {user_id}")
            print(f"   File: {filename}")
            print(f"   Challenge: {challenge}")
            
            # Download
            local_path = f"/tmp/{filename}"
            s3.download_file(config.BUCKET, filename, local_path)
            
            # Process
            detector = SimpleLiveness(local_path, challenge)
            success, movements, blink = detector.process()
            
            # Publish result
            result = {
                "user_id": user_id,
                "status": "IA_SUCCESS" if success else "IA_ERROR",
                "filename": filename,
                "movements_detected": movements,
                "blink_detected": blink,
                "timestamp": time.time()
            }
            
            r.publish('ia_results', json.dumps(result))
            
            # Cleanup
            if os.path.exists(local_path):
                os.remove(local_path)
        
        except Exception as e:
            print(f"\n❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    start_worker()
