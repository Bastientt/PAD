import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import os
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import time
import requests
import tempfile
import shutil
import boto3
from botocore.client import Config as BotoConfig

# ==================== CONFIGURATION ====================
class Config:
    MINIO_ENDPOINT = "" # ip minio
    MINIO_ACCESS_KEY = "access_key" # à modifier
    MINIO_SECRET_KEY = "secret_key" # à modifier
    MINIO_BUCKET_NAME = "videos-pad" #à modifier
    MINIO_SECURE = False # TRUE si HTTPS
    WATCH_INTERVAL = 10  
    
    LOCAL_FILE_OVERRIDE = None 

    SEUIL_SIMILARITE_BG = 0.85
    SEUIL_MOUVEMENT_HORIZONTAL = 10
    SEUIL_MOUVEMENT_VERTICAL = 12
    SEUIL_SIMILARITE_VETEMENTS = 0.70
    SEUIL_COULEUR_DOMINANTE = 0.65
    SEUIL_TEXTURE = 0.70
    
    ANALYSE_EVERY_N_FRAMES = 5
    DETECT_MOVEMENT_EVERY_N = 2
    RESIZE_SCALE = 0.5
    GRILLE_ZONES = 2
    HIST_BINS = 16
    
    MARGE_EXCLUSION_VISAGE = 200
    
    ZONE_VETEMENTS_RATIO = 0.25
    NB_COULEURS_DOMINANTES = 2
    
    MIN_DETECTION_CONFIDENCE = 0.5
    
    SMOOTHING_WINDOW = 4
    MIN_MOVEMENT_FRAMES = 5
    MIN_MOVEMENT_DISTANCE = 20
    SEUIL_POSITION_STABLE = 6
    MAX_GAP_BETWEEN_MOVEMENTS = 9
    CALIBRATION_FRAMES = 3
    
    SHOW_WINDOWS = True
    SAVE_REPORT = True
    
    USE_THREADING = True

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class VideoAnalyzer:
    def __init__(self, video_source, is_url=False):
        self.original_source_name = video_source
        self.is_url = is_url
        self.video_path = video_source
        self.temp_file = None
        
        self.cap = None
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE
        )
        
        if Config.USE_THREADING:
            self.executor = ThreadPoolExecutor(max_workers=2)
        else:
            self.executor = None
        
        self.frame_count = 0
        self.frames_analysees = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.start_time = None
        
        self.hist_references = None
        self.changements_bg = 0
        self.zones_instables = defaultdict(int)
        
        self.position_precedente = None
        self.direction_precedente = None
        self.frame_debut_mouvement = None
        self.mouvements_detectes = []
        self.historique_positions = []
        self.frames_consecutifs_direction = 0
        self.position_debut_mouvement = None
        self.distance_totale_mouvement = 0
        self.last_movement_frame = None
        self.frames_without_movement = 0
        
        self.position_neutre = None
        self.calibration_positions = []
        self.is_calibrated = False
        
        self.reference_couleurs = None
        self.reference_texture = None
        self.reference_clothing_hist = None
        self.changements_vetements = 0
        self.changements_couleurs = 0
        self.changements_texture = 0
        
        self.last_bg_stable = True
        self.last_zones_changees = []
        self.last_coherent = True
        self.last_sim_vet = 1.0
        self.last_sim_coul = 1.0
        self.last_sim_text = 1.0
        self.last_clothing_bbox = None
    
    def download_video(self):
        print(f"Téléchargement de la vidéo depuis : {self.original_source_name}")
        try:
            suffix = os.path.splitext(self.original_source_name)[1]
            if not suffix: suffix = '.mp4'
            
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            
            with requests.get(self.original_source_name, stream=True) as r:
                r.raise_for_status()
                with open(self.temp_file.name, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            
            self.video_path = self.temp_file.name
            print(f"Téléchargement terminé. Fichier temporaire : {self.video_path}")
            return True
        except Exception as e:
            print(f"Erreur lors du téléchargement : {e}")
            return False

    def load_video(self):
        if self.is_url:
            if not self.download_video():
                raise RuntimeError("Échec du téléchargement de la vidéo")

        print(f"Chargement du fichier : {self.video_path}")
        
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video non trouvee: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la video")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video chargee : {self.width}x{self.height} @ {self.fps}fps")
        print(f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)")
        print(f"\nOPTIMISATIONS:")
        print(f"  - Calibration: {Config.CALIBRATION_FRAMES} frames")
        print(f"  - Arriere-plan/Sujet: 1 frame/{Config.ANALYSE_EVERY_N_FRAMES}")
        print(f"  - Mouvements: 1 frame/{Config.DETECT_MOVEMENT_EVERY_N}")
        print(f"  - Seuil horizontal: {Config.SEUIL_MOUVEMENT_HORIZONTAL}px")
        print(f"  - Seuil vertical: {Config.SEUIL_MOUVEMENT_VERTICAL}px")
        print(f"  - Mouvement minimum: {Config.MIN_MOVEMENT_FRAMES} frames ({Config.MIN_MOVEMENT_FRAMES/self.fps*Config.DETECT_MOVEMENT_EVERY_N:.2f}s)")
        print(f"  - Distance minimale: {Config.MIN_MOVEMENT_DISTANCE}px")
        print(f"  - Gap toléré: {Config.MAX_GAP_BETWEEN_MOVEMENTS} frames")
        print(f"  - Redimensionnement: {Config.RESIZE_SCALE*100:.0f}%\n")
    
    def get_face_bbox(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]
        
        x_min = max(0, int(min(x_coords)))
        x_max = min(w, int(max(x_coords)))
        y_min = max(0, int(min(y_coords)))
        y_max = min(h, int(max(y_coords)))
        
        return (x_min, y_min, x_max, y_max)
    
    def create_analysis_zones(self, frame_shape, face_bbox):
        h, w = frame_shape[:2]
        zones = []
        
        grid_size = Config.GRILLE_ZONES
        zone_h = h // grid_size
        zone_w = w // grid_size
        
        fx_min, fy_min, fx_max, fy_max = face_bbox
        marge = Config.MARGE_EXCLUSION_VISAGE
        
        face_zone_expanded = (
            max(0, fx_min - marge),
            max(0, fy_min - marge),
            min(w, fx_max + marge),
            min(h, fy_max + marge)
        )
        
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = j * zone_w
                y1 = i * zone_h
                x2 = x1 + zone_w
                y2 = y1 + zone_h
                
                zone_center_x = (x1 + x2) // 2
                zone_center_y = (y1 + y2) // 2
                
                if not (face_zone_expanded[0] <= zone_center_x <= face_zone_expanded[2] and
                        face_zone_expanded[1] <= zone_center_y <= face_zone_expanded[3]):
                    zones.append((x1, y1, x2, y2))
        
        return zones
    
    def extract_histogram_from_zones(self, frame, zones):
        histograms = []
        
        for (x1, y1, x2, y2) in zones:
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    hist_b = cv2.calcHist([roi], [0], None, [Config.HIST_BINS], [0, 256])
                    hist_g = cv2.calcHist([roi], [1], None, [Config.HIST_BINS], [0, 256])
                    hist_r = cv2.calcHist([roi], [2], None, [Config.HIST_BINS], [0, 256])
                    
                    hist = np.concatenate([hist_b, hist_g, hist_r])
                    hist = hist.astype(np.float32)
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    
                    histograms.append(hist)
        
        return histograms if histograms else None
    
    def compare_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0.0
        
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        try:
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, similarity)
        except:
            return 0.0
    
    def analyze_background_stability(self, frame, face_bbox):
        zones = self.create_analysis_zones(frame.shape, face_bbox)
        
        if not zones:
            return True, []
        
        current_hists = self.extract_histogram_from_zones(frame, zones)
        
        if current_hists is None:
            return True, []
        
        if self.hist_references is None:
            self.hist_references = current_hists
            print(f"Reference arriere-plan etablie (frame {self.frame_count})")
            print(f"   {len(zones)} zones analysees\n")
            return True, []
        
        zones_changees = []
        for i, (hist_ref, hist_cur) in enumerate(zip(self.hist_references, current_hists)):
            similarite = self.compare_histograms(hist_ref, hist_cur)
            
            if similarite < Config.SEUIL_SIMILARITE_BG:
                zones_changees.append(i)
                self.zones_instables[i] += 1
        
        if zones_changees:
            self.changements_bg += 1
        
        return len(zones_changees) == 0, zones_changees
    
    def extract_clothing_region(self, frame, face_bbox):
        h, w = frame.shape[:2]
        fx_min, fy_min, fx_max, fy_max = face_bbox
        
        face_height = fy_max - fy_min
        clothing_top = min(h, fy_max + int(face_height * 0.2))
        clothing_height = int(h * Config.ZONE_VETEMENTS_RATIO)
        clothing_bottom = min(h, clothing_top + clothing_height)
        
        clothing_left = max(0, fx_min - 50)
        clothing_right = min(w, fx_max + 50)
        
        if clothing_bottom > clothing_top and clothing_right > clothing_left:
            roi = frame[clothing_top:clothing_bottom, clothing_left:clothing_right]
            return roi, (clothing_left, clothing_top, clothing_right, clothing_bottom)
        
        return None, None
    
    def extract_dominant_colors(self, roi, n_colors=2):
        if roi is None or roi.size == 0:
            return None
        
        if roi.shape[0] > 100 or roi.shape[1] > 100:
            scale = min(100 / roi.shape[0], 100 / roi.shape[1])
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        pixels = roi.reshape(-1, 3)
        
        if len(pixels) < n_colors:
            return None
        
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=3, max_iter=100)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(int)
        except:
            return None
    
    def calculate_texture_histogram(self, roi):
        if roi is None or roi.size == 0:
            return None
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        if gray.shape[0] > 100:
            scale = 100 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        
        hist = cv2.calcHist([magnitude.astype(np.float32)], [0], None, [64], [0, 256])
        
        hist = hist.astype(np.float32)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist
    
    def compare_colors(self, colors1, colors2):
        if colors1 is None or colors2 is None:
            return 0.0
        
        distances = []
        for c1 in colors1:
            min_dist = min([np.linalg.norm(c1 - c2) for c2 in colors2])
            distances.append(min_dist)
        
        avg_distance = np.mean(distances)
        max_distance = np.sqrt(3 * 255**2)
        similarity = 1 - (avg_distance / max_distance)
        
        return max(0.0, similarity)
    
    def analyze_subject_consistency(self, frame, face_bbox):
        clothing_roi, clothing_bbox = self.extract_clothing_region(frame, face_bbox)
        
        if clothing_roi is None:
            return True, 1.0, 1.0, 1.0, None
        
        current_colors = self.extract_dominant_colors(clothing_roi, Config.NB_COULEURS_DOMINANTES)
        current_texture = self.calculate_texture_histogram(clothing_roi)
        
        if self.reference_couleurs is None:
            self.reference_couleurs = current_colors
            self.reference_texture = current_texture
            print(f"Reference sujet etablie (frame {self.frame_count})")
            if current_colors is not None:
                print(f"   Couleurs dominantes: {current_colors}\n")
            return True, 1.0, 1.0, 1.0, clothing_bbox
        
        similarite_couleur = self.compare_colors(self.reference_couleurs, current_colors)
        similarite_texture = self.compare_histograms(self.reference_texture, current_texture)
        
        coherent = True
        
        if similarite_couleur < Config.SEUIL_COULEUR_DOMINANTE:
            self.changements_couleurs += 1
            coherent = False
        
        if similarite_texture < Config.SEUIL_TEXTURE:
            self.changements_texture += 1
            coherent = False
        
        return coherent, 1.0, similarite_couleur, similarite_texture, clothing_bbox
    
    def get_head_position(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        nose_tip = face_landmarks.landmark[4]
        nose_x = nose_tip.x * w
        nose_y = nose_tip.y * h
        
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        eye_left_x = left_eye.x * w
        eye_right_x = right_eye.x * w
        eye_center_x = (eye_left_x + eye_right_x) / 2
        
        eye_left_y = left_eye.y * h
        eye_right_y = right_eye.y * h
        eye_center_y = (eye_left_y + eye_right_y) / 2
        
        chin = face_landmarks.landmark[152]
        chin_y = chin.y * h
        
        return {
            'nose_x': nose_x,
            'nose_y': nose_y,
            'eye_center_x': eye_center_x,
            'eye_center_y': eye_center_y,
            'chin_y': chin_y,
            'horizontal_offset': nose_x - eye_center_x,
            'vertical_offset': nose_y - eye_center_y,
            'nose_chin_distance': chin_y - nose_y
        }
    
    def calibrate_neutral_position(self, position):
        self.calibration_positions.append(position)
        
        if len(self.calibration_positions) >= Config.CALIBRATION_FRAMES:
            self.position_neutre = {
                'horizontal_offset': np.mean([p['horizontal_offset'] for p in self.calibration_positions]),
                'vertical_offset': np.mean([p['vertical_offset'] for p in self.calibration_positions]),
                'nose_chin_distance': np.mean([p['nose_chin_distance'] for p in self.calibration_positions])
            }
            self.is_calibrated = True
            print(f"Calibration terminee (frame {self.frame_count})")
            print(f"   Position neutre: H={self.position_neutre['horizontal_offset']:.1f}px, V={self.position_neutre['vertical_offset']:.1f}px\n")
    
    def smooth_position(self, position):
        self.historique_positions.append(position)
        
        if len(self.historique_positions) > Config.SMOOTHING_WINDOW:
            self.historique_positions.pop(0)
        
        avg_horizontal = np.mean([p['horizontal_offset'] for p in self.historique_positions])
        avg_vertical = np.mean([p['vertical_offset'] for p in self.historique_positions])
        avg_nose_chin = np.mean([p['nose_chin_distance'] for p in self.historique_positions])
        
        return {
            'horizontal_offset': avg_horizontal,
            'vertical_offset': avg_vertical,
            'nose_chin_distance': avg_nose_chin
        }
    
    def detect_head_movement(self, face_landmarks, frame_shape):
        position = self.get_head_position(face_landmarks, frame_shape)
        
        if not self.is_calibrated:
            self.calibrate_neutral_position(position)
            return None, 0, position
        
        position_lissee = self.smooth_position(position)
        
        horizontal_offset = position_lissee['horizontal_offset'] - self.position_neutre['horizontal_offset']
        vertical_offset = position_lissee['vertical_offset'] - self.position_neutre['vertical_offset']
        
        distance_h = abs(horizontal_offset)
        distance_v = abs(vertical_offset)
        
        if distance_h < Config.SEUIL_POSITION_STABLE and distance_v < Config.SEUIL_POSITION_STABLE:
            return None, 0, position_lissee
        
        direction = None
        distance = 0
        
        if distance_h > Config.SEUIL_MOUVEMENT_HORIZONTAL and distance_h > distance_v:
            if horizontal_offset > 0:
                direction = "DROITE"
            else:
                direction = "GAUCHE"
            distance = distance_h
            
        elif distance_v > Config.SEUIL_MOUVEMENT_VERTICAL:
            if vertical_offset > 0:
                direction = "BAS"
            else:
                direction = "HAUT"
            distance = distance_v
        
        return direction, distance, position_lissee
    
    def track_movement_sequence(self, direction, frame_num, position):
        if direction == self.direction_precedente:
            self.frames_consecutifs_direction += 1
            self.frames_without_movement = 0
            
            if self.position_debut_mouvement and self.position_neutre:
                if direction in ['GAUCHE', 'DROITE']:
                    offset_debut = self.position_debut_mouvement['horizontal_offset'] - self.position_neutre['horizontal_offset']
                    offset_actuel = position['horizontal_offset'] - self.position_neutre['horizontal_offset']
                    dist = abs(offset_actuel - offset_debut)
                else:
                    offset_debut = self.position_debut_mouvement['vertical_offset'] - self.position_neutre['vertical_offset']
                    offset_actuel = position['vertical_offset'] - self.position_neutre['vertical_offset']
                    dist = abs(offset_actuel - offset_debut)
                
                self.distance_totale_mouvement = max(self.distance_totale_mouvement, dist)
                
        elif direction is None and self.direction_precedente is not None:
            self.frames_without_movement += 1
            
            if self.frames_without_movement >= Config.MAX_GAP_BETWEEN_MOVEMENTS:
                self._finalize_movement(frame_num)
                
        else:
            if self.direction_precedente is not None:
                self._finalize_movement(frame_num)
            
            if direction is not None:
                self.direction_precedente = direction
                self.frame_debut_mouvement = frame_num
                self.frames_consecutifs_direction = 1
                self.position_debut_mouvement = position.copy()
                self.distance_totale_mouvement = 0
                self.frames_without_movement = 0
    
    def _finalize_movement(self, frame_num):
        if self.frame_debut_mouvement is None:
            return
            
        duree_frames = frame_num - self.frame_debut_mouvement - (self.frames_without_movement * Config.DETECT_MOVEMENT_EVERY_N)
        
        if (duree_frames >= Config.MIN_MOVEMENT_FRAMES and 
            self.distance_totale_mouvement >= Config.MIN_MOVEMENT_DISTANCE):
            
            mouvement = {
                'direction': self.direction_precedente,
                'debut': self.frame_debut_mouvement,
                'fin': frame_num - (self.frames_without_movement * Config.DETECT_MOVEMENT_EVERY_N),
                'distance': self.distance_totale_mouvement
            }
            self.mouvements_detectes.append(mouvement)
        
        self.direction_precedente = None
        self.frame_debut_mouvement = None
        self.frames_consecutifs_direction = 0
        self.position_debut_mouvement = None
        self.distance_totale_mouvement = 0
        self.frames_without_movement = 0
    
    def draw_analysis(self, frame, face_bbox, direction, distance):
        display = frame.copy()
        
        fx_min, fy_min, fx_max, fy_max = face_bbox
        color = (0, 255, 0) if (self.last_bg_stable and self.last_coherent) else (0, 165, 255)
        cv2.rectangle(display, (fx_min, fy_min), (fx_max, fy_max), color, 2)
        
        if self.last_clothing_bbox:
            cx1, cy1, cx2, cy2 = self.last_clothing_bbox
            cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (255, 200, 0), 2)
        
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        cv2.putText(display, f"Frame: {self.frame_count}/{self.total_frames}", 
                   (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += 30
        
        if not self.is_calibrated:
            calib_text = f"CALIBRATION: {len(self.calibration_positions)}/{Config.CALIBRATION_FRAMES}"
            cv2.putText(display, calib_text, 
                       (10, y_offset), font, font_scale, (0, 255, 255), thickness)
            y_offset += 30
        
        bg_text = "STABLE" if self.last_bg_stable else f"INSTABLE"
        bg_color = (0, 255, 0) if self.last_bg_stable else (0, 0, 255)
        cv2.putText(display, f"BG: {bg_text}", 
                   (10, y_offset), font, font_scale, bg_color, thickness)
        y_offset += 30
        
        sujet_text = "COHERENT" if self.last_coherent else "INCOHERENT"
        sujet_color = (0, 255, 0) if self.last_coherent else (0, 0, 255)
        cv2.putText(display, f"Sujet: {sujet_text}", 
                   (10, y_offset), font, font_scale, sujet_color, thickness)
        y_offset += 25
        
        cv2.putText(display, f"Couleur: {self.last_sim_coul:.2f}", 
                   (10, y_offset), font, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        cv2.putText(display, f"Texture: {self.last_sim_text:.2f}", 
                   (10, y_offset), font, 0.5, (200, 200, 200), 1)
        y_offset += 30
        
        if direction and self.is_calibrated:
            mvt_color = (255, 0, 255)
            cv2.putText(display, f">>> {direction} ({distance:.1f}px)", 
                       (10, y_offset), font, font_scale, mvt_color, thickness)
            y_offset += 25
            if self.distance_totale_mouvement > 0:
                cv2.putText(display, f"Distance totale: {self.distance_totale_mouvement:.1f}px", 
                           (10, y_offset), font, 0.5, (200, 200, 200), 1)
        elif self.frames_without_movement > 0 and self.is_calibrated:
            cv2.putText(display, f"Gap: {self.frames_without_movement}/{Config.MAX_GAP_BETWEEN_MOVEMENTS//Config.DETECT_MOVEMENT_EVERY_N}", 
                       (10, y_offset), font, 0.5, (255, 255, 0), 1)
        elif self.is_calibrated:
            cv2.putText(display, f"Position: STABLE", 
                       (10, y_offset), font, 0.5, (150, 150, 150), 1)
        y_offset += 30
        
        cv2.putText(display, f"Mouvements valides: {len(self.mouvements_detectes)}", 
                   (10, y_offset), font, 0.5, (200, 200, 200), 1)
        
        return display
    
    def process(self):
        print("Analyse en cours...\n")
        self.start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.direction_precedente is not None:
                    self._finalize_movement(self.frame_count)
                break
            
            self.frame_count += 1
            
            if Config.RESIZE_SCALE < 1.0:
                frame_resized = cv2.resize(frame, None, fx=Config.RESIZE_SCALE, 
                                          fy=Config.RESIZE_SCALE, 
                                          interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
            
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                continue
            
            face_landmarks = results.multi_face_landmarks[0]
            face_bbox = self.get_face_bbox(face_landmarks, frame_resized.shape)
            
            direction = None
            distance = 0
            position = None
            if self.frame_count % Config.DETECT_MOVEMENT_EVERY_N == 0:
                direction, distance, position = self.detect_head_movement(face_landmarks, frame_resized.shape)
                if self.is_calibrated:
                    self.track_movement_sequence(direction, self.frame_count, position)
            
            if self.frame_count % Config.ANALYSE_EVERY_N_FRAMES == 0:
                self.frames_analysees += 1
                
                if Config.USE_THREADING and self.executor:
                    future_bg = self.executor.submit(self.analyze_background_stability, 
                                                    frame_resized, face_bbox)
                    future_subject = self.executor.submit(self.analyze_subject_consistency, 
                                                         frame_resized, face_bbox)
                    
                    bg_stable, zones_changees = future_bg.result()
                    coherent, sim_vet, sim_coul, sim_text, clothing_bbox = future_subject.result()
                else:
                    bg_stable, zones_changees = self.analyze_background_stability(frame_resized, face_bbox)
                    coherent, sim_vet, sim_coul, sim_text, clothing_bbox = self.analyze_subject_consistency(frame_resized, face_bbox)
                
                self.last_bg_stable = bg_stable
                self.last_zones_changees = zones_changees
                self.last_coherent = coherent
                self.last_sim_vet = sim_vet
                self.last_sim_coul = sim_coul
                self.last_sim_text = sim_text
                self.last_clothing_bbox = clothing_bbox
            
            if Config.SHOW_WINDOWS:
                display = self.draw_analysis(frame_resized, face_bbox, direction, distance)
                
                h, w = display.shape[:2]
                scale = 800 / w if w > 800 else 1.0
                if scale != 1.0:
                    display = cv2.resize(display, (int(w*scale), int(h*scale)))
                
                cv2.imshow('Analyse PAD', display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        elapsed_time = time.time() - self.start_time
        print(f"\nTemps d'analyse: {elapsed_time:.2f}s")
        print(f"Vitesse: {self.frame_count/elapsed_time:.1f} fps (traitement)")
    
    def generate_report(self):
        print("\n" + "="*70)
        print("RAPPORT D'ANALYSE PAD")
        print("="*70)
        
        taux_stabilite_bg = ((self.frames_analysees - self.changements_bg) / self.frames_analysees * 100) if self.frames_analysees > 0 else 0
        
        total_changements_sujet = (self.changements_vetements + 
                                   self.changements_couleurs + 
                                   self.changements_texture)
        taux_coherence = ((self.frames_analysees - total_changements_sujet) / self.frames_analysees * 100) if self.frames_analysees > 0 else 0
        
        validation_bg = taux_stabilite_bg >= 80
        validation_sujet = taux_coherence >= 80
        validation_globale = validation_bg and validation_sujet
        
        print(f"\nVIDEO: {self.original_source_name}")
        print(f"Total frames: {self.total_frames}")
        print(f"Frames analysees (BG/Sujet): {self.frames_analysees}")
        print(f"Frames analysees (Mouvements): {self.frame_count // Config.DETECT_MOVEMENT_EVERY_N}")
        print(f"Duree: {self.total_frames/self.fps:.2f}s")
        
        print(f"\n{'─'*70}")
        print(f"MOUVEMENTS DE TETE DETECTES: {len(self.mouvements_detectes)}")
        print(f"{'─'*70}")
        if self.mouvements_detectes:
            for i, mvt in enumerate(self.mouvements_detectes, 1):
                duree_frames = mvt['fin'] - mvt['debut']
                duree_sec = duree_frames / self.fps
                dist = mvt.get('distance', 0)
                print(f"{i:2}. {mvt['direction']:>6} | Frames {mvt['debut']:>5}-{mvt['fin']:>5} | {duree_sec:>5.2f}s | {dist:>5.1f}px")
        else:
            print("   Aucun mouvement valide détecté")
        
        print(f"\n{'─'*70}")
        print(f"ARRIERE-PLAN:")
        print(f"  Changements: {self.changements_bg}")
        print(f"  Stabilite: {taux_stabilite_bg:.1f}%")
        print(f"  Validation: {'✓ OK' if validation_bg else '✗ FAIL'}")
        
        print(f"\nSUJET:")
        print(f"  Changements couleurs: {self.changements_couleurs}")
        print(f"  Changements texture: {self.changements_texture}")
        print(f"  Coherence: {taux_coherence:.1f}%")
        print(f"  Validation: {'✓ OK' if validation_sujet else '✗ FAIL'}")
        
        print(f"\n{'='*70}")
        print(f"VALIDATION GLOBALE: {'✓✓✓ VIDEO AUTHENTIQUE' if validation_globale else '✗✗✗ VIDEO SUSPECTE'}")
        print(f"{'='*70}")
        
        if Config.SAVE_REPORT:
            safe_name = os.path.basename(self.original_source_name)
            if self.is_url:
                 safe_name = "".join([c for c in safe_name if c.isalnum() or c in (' ', '.', '_')]).rstrip()
                 if not safe_name: safe_name = "video_url"
            
            report_filename = f'analyse_report_{safe_name}.txt'
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("RAPPORT D'ANALYSE PAD\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"VIDEO: {self.original_source_name}\n")
                f.write(f"Total frames: {self.total_frames}\n")
                f.write(f"Frames analysees (BG/Sujet): {self.frames_analysees}\n")
                f.write(f"Frames analysees (Mouvements): {self.frame_count // Config.DETECT_MOVEMENT_EVERY_N}\n")
                f.write(f"Duree: {self.total_frames/self.fps:.2f}s\n\n")
                
                f.write(f"MOUVEMENTS: {len(self.mouvements_detectes)}\n")
                f.write("─"*70 + "\n")
                for i, mvt in enumerate(self.mouvements_detectes, 1):
                    duree_frames = mvt['fin'] - mvt['debut']
                    duree_sec = duree_frames / self.fps
                    dist = mvt.get('distance', 0)
                    f.write(f"{i:2}. {mvt['direction']:>6} | Frames {mvt['debut']:>5}-{mvt['fin']:>5} | {duree_sec:>5.2f}s | {dist:>5.1f}px\n")
                
                f.write(f"\nARRIERE-PLAN:\n")
                f.write(f"Changements: {self.changements_bg}\n")
                f.write(f"Stabilite: {taux_stabilite_bg:.1f}%\n")
                f.write(f"Validation: {'OK' if validation_bg else 'FAIL'}\n")
                
                f.write(f"\nSUJET:\n")
                f.write(f"Changements couleurs: {self.changements_couleurs}\n")
                f.write(f"Changements texture: {self.changements_texture}\n")
                f.write(f"Coherence: {taux_coherence:.1f}%\n")
                f.write(f"Validation: {'OK' if validation_sujet else 'FAIL'}\n")
                
                f.write(f"\nVALIDATION GLOBALE: {'OK' if validation_globale else 'FAIL'}\n")
            
            print(f"\nRapport sauvegarde dans '{report_filename}'")
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.executor:
            self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()
        
        if self.temp_file and os.path.exists(self.temp_file.name):
            try:
                self.temp_file.close()
                os.unlink(self.temp_file.name)
                print("Fichier temporaire supprimé.")
            except Exception as e:
                print(f"Erreur suppression fichier temp: {e}")

def main():
    s3_client = boto3.client(
        's3',
        endpoint_url=f"{'https' if Config.MINIO_SECURE else 'http'}://{Config.MINIO_ENDPOINT}",
        aws_access_key_id=Config.MINIO_ACCESS_KEY,
        aws_secret_access_key=Config.MINIO_SECRET_KEY,
        config=BotoConfig(signature_version='s3v4'),
        region_name='us-east-1'
    )

    deja_traites = set()
    print(f"--- Surveillance active du bucket : {Config.MINIO_BUCKET_NAME} ---")

    try:
        while True:
            response = s3_client.list_objects_v2(Bucket=Config.MINIO_BUCKET_NAME)
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    file_key = obj['Key']
                    
                    if file_key not in deja_traites and file_key.lower().endswith(('.mp4', '.mov', '.avi')):
                        print(f"\n[NOUVEAU FICHIER] Détection de : {file_key}")
                        
                        url_presignee = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': Config.MINIO_BUCKET_NAME, 'Key': file_key},
                            ExpiresIn=3600
                        )

                        analyzer = VideoAnalyzer(url_presignee, is_url=True)
                        
                        try:
                            analyzer.load_video()
                            analyzer.process()
                            analyzer.generate_report()
                            deja_traites.add(file_key)  # Marquer comme fait
                        except Exception as e:
                            print(f"Erreur lors du traitement de {file_key}: {e}")
                        finally:
                            analyzer.cleanup()
            
            time.sleep(Config.WATCH_INTERVAL)

    except KeyboardInterrupt:
        print("\nArrêt manuel du script.")

if __name__ == "__main__":
    main()