import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import os

# ==================== CONFIGURATION ====================
class Config:
    VIDEO_SOURCE = 'IMG_0006.MOV'
    
    # Seuils
    SEUIL_SIMILARITE_BG = 0.75  # Plus permissif
    SEUIL_MOUVEMENT = 15
    
    # Zones d'analyse
    GRILLE_ZONES = 3  # Grille 3x3 = 9 zones
    MARGE_EXCLUSION_VISAGE = 250  # pixels autour du visage à ignorer
    
    # Détection
    MIN_DETECTION_CONFIDENCE = 0.7
    ANALYSE_EVERY_N_FRAMES = 5  # Analyser 1 frame sur 5 (plus rapide)
    
    # Affichage
    SHOW_WINDOWS = True
    SAVE_REPORT = True

# ==================== INITIALISATION MEDIAPIPE ====================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class VideoAnalyzer:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = None
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE
        )
        
        # Statistiques
        self.frame_count = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        
        # Arrière-plan
        self.hist_references = None  # Liste d'histogrammes (un par zone)
        self.changements_bg = 0
        self.zones_instables = defaultdict(int)
        
        # Mouvements
        self.direction_precedente = None
        self.frame_debut_mouvement = None
        self.mouvements = []
    
    def load_video(self):
        """Charge la vidéo (locale ou URL)"""
        print("⏳ Chargement de la vidéo...")
        
        if self.video_source.startswith('http'):
            import requests
            response = requests.get(self.video_source, stream=True)
            with open('temp_video.mp4', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.cap = cv2.VideoCapture('temp_video.mp4')
        else:
            self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            raise Exception(f"Impossible d'ouvrir : {self.video_source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✅ Vidéo chargée : {self.width}x{self.height} @ {self.fps}fps")
        print(f"📊 {self.total_frames} frames ({self.total_frames/self.fps:.1f}s)\n")
    
    def get_face_bbox(self, landmarks):
        """Obtient la bounding box du visage avec marge"""
        xs = [lm.x * self.width for lm in landmarks]
        ys = [lm.y * self.height for lm in landmarks]
        
        x_min = max(0, int(min(xs)) - Config.MARGE_EXCLUSION_VISAGE)
        x_max = min(self.width, int(max(xs)) + Config.MARGE_EXCLUSION_VISAGE)
        y_min = max(0, int(min(ys)) - Config.MARGE_EXCLUSION_VISAGE)
        y_max = min(self.height, int(max(ys)) + Config.MARGE_EXCLUSION_VISAGE)
        
        return x_min, y_min, x_max, y_max
    
    def detect_head_direction(self, landmarks):
        """Détermine la direction du regard"""
        nez = landmarks[1]
        menton = landmarks[152]
        oeil_gauche = landmarks[33]
        oeil_droit = landmarks[263]
        
        nez_x = nez.x * self.width
        nez_y = nez.y * self.height
        centre_yeux_x = (oeil_gauche.x + oeil_droit.x) / 2 * self.width
        centre_yeux_y = (oeil_gauche.y + oeil_droit.y) / 2 * self.height
        
        decalage_x = nez_x - centre_yeux_x
        decalage_y = nez_y - centre_yeux_y
        
        if abs(decalage_x) > abs(decalage_y):
            if decalage_x > Config.SEUIL_MOUVEMENT:
                return "Droite"
            elif decalage_x < -Config.SEUIL_MOUVEMENT:
                return "Gauche"
        else:
            if decalage_y > Config.SEUIL_MOUVEMENT:
                return "Bas"
            elif decalage_y < -Config.SEUIL_MOUVEMENT:
                return "Haut"
        
        return "Centre"
    
    def analyze_background_grille(self, frame, face_bbox):
        """Analyse l'arrière-plan avec grille 3x3 (excluant le visage)"""
        h, w = frame.shape[:2]
        n = Config.GRILLE_ZONES
        
        zone_h = h // n
        zone_w = w // n
        
        histogrammes = []
        masque_zones = np.zeros((h, w), dtype=np.uint8)
        zones_analysees = 0
        
        face_x_min, face_y_min, face_x_max, face_y_max = face_bbox
        
        for i in range(n):
            for j in range(n):
                # Coordonnées de la zone
                y1 = i * zone_h
                y2 = (i + 1) * zone_h if i < n - 1 else h
                x1 = j * zone_w
                x2 = (j + 1) * zone_w if j < n - 1 else w
                
                # Vérifier si zone chevauche le visage
                chevauchement_x = not (x2 < face_x_min or x1 > face_x_max)
                chevauchement_y = not (y2 < face_y_min or y1 > face_y_max)
                
                if chevauchement_x and chevauchement_y:
                    # Zone touche le visage → ignorer
                    continue
                
                # Zone valide
                zones_analysees += 1
                masque_zones[y1:y2, x1:x2] = 255
                
                # Extraire la zone
                zone = frame[y1:y2, x1:x2]
                zone_gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
                
                # Calculer histogramme
                hist = cv2.calcHist([zone_gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                histogrammes.append(hist)
        
        pixels_zones = np.count_nonzero(masque_zones)
        pourcentage_zones = (pixels_zones / (h * w)) * 100
        
        changement = False
        similarite_moyenne = 1.0
        
        if self.hist_references is None:
            self.hist_references = histogrammes
            print(f"📌 Référence arrière-plan établie (frame {self.frame_count})")
            print(f"   └─ {zones_analysees} zones analysées ({pourcentage_zones:.1f}% de l'image)\n")
        else:
            if len(histogrammes) != len(self.hist_references):
                # Nombre de zones différent (visage a bougé beaucoup)
                return False, 1.0, masque_zones
            
            similarites = []
            for idx, (hist_ref, hist_cur) in enumerate(zip(self.hist_references, histogrammes)):
                sim = cv2.compareHist(hist_ref, hist_cur, cv2.HISTCMP_CORREL)
                similarites.append(sim)
                
                if sim < Config.SEUIL_SIMILARITE_BG:
                    self.zones_instables[idx] += 1
            
            similarite_moyenne = np.mean(similarites)
            
            if similarite_moyenne < Config.SEUIL_SIMILARITE_BG:
                changement = True
                self.changements_bg += 1
                print(f"⚠️  Frame {self.frame_count}: Changement arrière-plan (sim moy: {similarite_moyenne:.3f})")
        
        return changement, similarite_moyenne, masque_zones
    
    def enregistrer_mouvement(self, direction):
        """Enregistre un mouvement quand la direction change"""
        if direction != self.direction_precedente:
            if self.direction_precedente is not None and self.direction_precedente != "Centre":
                self.mouvements.append({
                    'direction': self.direction_precedente,
                    'debut': self.frame_debut_mouvement,
                    'fin': self.frame_count - Config.ANALYSE_EVERY_N_FRAMES
                })
                print(f"✅ Mouvement '{self.direction_precedente}': frames {self.frame_debut_mouvement}-{self.frame_count - Config.ANALYSE_EVERY_N_FRAMES}")
            
            if direction != "Centre":
                self.frame_debut_mouvement = self.frame_count
            
            self.direction_precedente = direction
    
    def process(self):
        """Traitement principal"""
        print("🚀 Analyse en cours...\n")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            if self.frame_count % Config.ANALYSE_EVERY_N_FRAMES != 0:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Détecter direction
                direction = self.detect_head_direction(landmarks)
                self.enregistrer_mouvement(direction)
                
                # Obtenir bbox du visage
                face_bbox = self.get_face_bbox(landmarks)
                
                # Analyser arrière-plan (grille sans visage)
                changement, similarite, masque_zones = self.analyze_background_grille(frame, face_bbox)
                
                if Config.SHOW_WINDOWS:
                    # Dessiner mesh
                    mp_drawing.draw_landmarks(
                        frame,
                        results.multi_face_landmarks[0],
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Dessiner bbox visage exclu (rouge)
                    x_min, y_min, x_max, y_max = face_bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    
                    # Infos
                    cv2.putText(frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Direction: {direction}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"BG Sim: {similarite:.3f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if similarite > Config.SEUIL_SIMILARITE_BG else (0, 0, 255), 2)
                    
                    # Visualiser zones analysées (vert)
                    overlay = frame.copy()
                    overlay[masque_zones == 255] = overlay[masque_zones == 255] * 0.7 + np.array([0, 255, 0]) * 0.3
                    
                    cv2.imshow('1. Video + Detection', frame)
                    cv2.imshow('2. Zones Analysees (Vert)', overlay)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)
            else:
                print(f"⚠️  Frame {self.frame_count}: Aucun visage détecté")
        
        if self.direction_precedente is not None and self.direction_precedente != "Centre":
            self.mouvements.append({
                'direction': self.direction_precedente,
                'debut': self.frame_debut_mouvement,
                'fin': self.frame_count
            })
    
    def generate_report(self):
        """Génère le rapport final"""
        print(f"\n{'='*70}")
        print(f"📊 RAPPORT D'ANALYSE")
        print(f"{'='*70}")
        print(f"📹 Source           : {self.video_source}")
        print(f"📏 Résolution       : {self.width}x{self.height}")
        print(f"⏱️  Durée            : {self.total_frames/self.fps:.2f}s ({self.total_frames} frames)")
        print(f"🎯 Frames analysées : {self.frame_count} (1 sur {Config.ANALYSE_EVERY_N_FRAMES})")
        
        print(f"\n{'─'*70}")
        print(f"🎭 MOUVEMENTS DE TÊTE DÉTECTÉS : {len(self.mouvements)}")
        print(f"{'─'*70}")
        
        if self.mouvements:
            for i, mvt in enumerate(self.mouvements, 1):
                duree_sec = (mvt['fin'] - mvt['debut']) / self.fps
                print(f"{i:2}. {mvt['direction']:>6} | Frames {mvt['debut']:>4}-{mvt['fin']:>4} | Durée: {duree_sec:.2f}s")
        else:
            print("   Aucun mouvement détecté")
        
        print(f"\n{'─'*70}")
        print(f"🖼️  STABILITÉ ARRIÈRE-PLAN")
        print(f"{'─'*70}")
        print(f"⚠️  Changements détectés : {self.changements_bg}")
        
        frames_analyses = self.frame_count // Config.ANALYSE_EVERY_N_FRAMES
        taux_stabilite = ((frames_analyses - self.changements_bg) / frames_analyses) * 100 if frames_analyses > 0 else 0
        print(f"📈 Taux de stabilité    : {taux_stabilite:.1f}%")
        
        if self.changements_bg == 0:
            print(f"✅ Arrière-plan STABLE - Validation réussie ✓")
        elif self.changements_bg <= 3:
            print(f"⚠️  Arrière-plan LÉGÈREMENT INSTABLE (tolérable)")
        else:
            print(f"❌ Arrière-plan INSTABLE - Validation échouée ✗")
        
        # Zones les plus instables
        if self.zones_instables:
            print(f"\n📍 Zones les plus instables:")
            zones_triees = sorted(self.zones_instables.items(), key=lambda x: x[1], reverse=True)[:3]
            for zone_idx, count in zones_triees:
                print(f"   └─ Zone {zone_idx}: {count} changements")
        
        print(f"{'='*70}\n")
        
        if Config.SAVE_REPORT:
            with open('analyse_report.txt', 'w', encoding='utf-8') as f:
                f.write(f"RAPPORT D'ANALYSE VIDÉO\n")
                f.write(f"{'='*70}\n")
                f.write(f"Source: {self.video_source}\n")
                f.write(f"Frames analysées: {frames_analyses}\n")
                f.write(f"Mouvements: {len(self.mouvements)}\n")
                f.write(f"Changements BG: {self.changements_bg}\n")
                f.write(f"Stabilité: {taux_stabilite:.1f}%\n")
                f.write(f"\nMOUVEMENTS DÉTAILLÉS:\n")
                for mvt in self.mouvements:
                    duree_sec = (mvt['fin'] - mvt['debut']) / self.fps
                    f.write(f"- {mvt['direction']} (frames {mvt['debut']}-{mvt['fin']}, {duree_sec:.2f}s)\n")
            print("💾 Rapport sauvegardé dans 'analyse_report.txt'")
    
    def cleanup(self):
        """Nettoyage"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if os.path.exists('temp_video.mp4'):
            os.remove('temp_video.mp4')

# ==================== MAIN ====================
def main():
    analyzer = VideoAnalyzer(Config.VIDEO_SOURCE)
    
    try:
        analyzer.load_video()
        analyzer.process()
        analyzer.generate_report()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main()
