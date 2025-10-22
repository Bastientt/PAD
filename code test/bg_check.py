import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
video_path = 'IMG_0006.MOV'
seuil_similarite_bg = 0.85
bordure_analyse = 100  # Largeur de la bordure à analyser (en pixels)

# ==================== CONTOURNER PYTORCH 2.6 ====================
print("⏳ Chargement du modèle YOLO...")

original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

try:
    model = YOLO('yolov8n.pt')
    print("✅ Modèle YOLO chargé\n")
finally:
    torch.load = original_load

# ==================== CHARGEMENT VIDÉO ====================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Impossible d'ouvrir : {video_path}")
    exit()

# Obtenir les dimensions de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📹 Vidéo : {width}x{height} @ {fps}fps ({total_frames} frames)")
print(f"🔍 Analyse des bordures de {bordure_analyse}px\n")

# ==================== VARIABLES ====================
hist_reference = None
frame_count = 0
changements_detectes = 0
personne_detectee_count = 0

print("📹 Analyse en cours...\n")

# ==================== BOUCLE PRINCIPALE ====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # ========== DÉTECTION YOLO ==========
    results = model(frame, verbose=False, conf=0.5)
    
    # Créer un masque pour les personnes (inversé pour l'arrière-plan)
    masque_personne = np.zeros((h, w), dtype=np.uint8)
    
    personne_detectee = False
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            
            # Si c'est une personne (classe 0)
            if cls == 0:
                personne_detectee = True
                personne_detectee_count += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # RÉDUIRE la bounding box de 10% pour éviter les bords
                marge_x = int((x2 - x1) * 0.10)
                marge_y = int((y2 - y1) * 0.10)
                
                x1_reduit = min(x2 - 10, x1 + marge_x)
                y1_reduit = min(y2 - 10, y1 + marge_y)
                x2_reduit = max(x1 + 10, x2 - marge_x)
                y2_reduit = max(y1 + 10, y2 - marge_y)
                
                # Remplir le masque avec la zone réduite
                masque_personne[y1_reduit:y2_reduit, x1_reduit:x2_reduit] = 255
                
                # Dessiner les rectangles
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # YOLO (bleu)
                cv2.rectangle(frame, (x1_reduit, y1_reduit), (x2_reduit, y2_reduit), (0, 255, 0), 2)  # Masque (vert)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ========== CRÉER MASQUE DES BORDS ==========
    masque_bords = np.zeros((h, w), dtype=np.uint8)
    
    # Haut
    masque_bords[0:bordure_analyse, :] = 255
    # Bas
    masque_bords[h-bordure_analyse:h, :] = 255
    # Gauche
    masque_bords[:, 0:bordure_analyse] = 255
    # Droite
    masque_bords[:, w-bordure_analyse:w] = 255
    
    # Visualiser les zones analysées
    frame_zones = frame.copy()
    frame_zones[masque_bords == 255] = frame_zones[masque_bords == 255] * 0.5 + np.array([0, 255, 255]) * 0.5

    # ========== MASQUE ARRIÈRE-PLAN (bords sans personne) ==========
    masque_bg = cv2.bitwise_not(masque_personne)  # Inverser
    masque_bg_bords = cv2.bitwise_and(masque_bg, masque_bords)  # Seulement les bords
    
    # Extraire l'arrière-plan des bords
    arriere_plan = cv2.bitwise_and(frame, frame, mask=masque_bg_bords)
    arriere_plan_gray = cv2.cvtColor(arriere_plan, cv2.COLOR_BGR2GRAY)

    # Calculer le pourcentage de masque
    pixels_masque = np.count_nonzero(masque_personne)
    pourcentage_masque = (pixels_masque / (h * w)) * 100
    
    pixels_bords_valides = np.count_nonzero(masque_bg_bords)
    pourcentage_bords = (pixels_bords_valides / (h * w)) * 100

    # ========== COMPARAISON HISTOGRAMME ==========
    hist_actuel = cv2.calcHist([arriere_plan_gray], [0], masque_bg_bords, [256], [0, 256])
    hist_actuel = cv2.normalize(hist_actuel, hist_actuel).flatten()

    changement = False
    similarite = 0.0

    if hist_reference is None:
        hist_reference = hist_actuel
        print(f"📌 Référence établie (frame {frame_count})")
        print(f"   └─ Zone d'analyse : {pourcentage_bords:.1f}% de l'image (bords)\n")
    else:
        similarite = cv2.compareHist(hist_reference, hist_actuel, cv2.HISTCMP_CORREL)
        
        if similarite < seuil_similarite_bg:
            changement = True
            changements_detectes += 1
            print(f"⚠️  Frame {frame_count} : Changement détecté ! (similarité: {similarite:.3f})")

    # ========== AFFICHAGE ==========
    # Texte d'information
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Similarite BG: {similarite:.3f}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if similarite > seuil_similarite_bg else (0, 0, 255), 2)
    
    cv2.putText(frame, f"Masque personne: {pourcentage_masque:.1f}%", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.putText(frame, f"Zone bords analysee: {pourcentage_bords:.1f}%", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.putText(frame, f"Changements: {changements_detectes}", 
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    status = "✓ Personne detectee" if personne_detectee else "✗ Aucune personne"
    couleur_status = (0, 255, 0) if personne_detectee else (0, 0, 255)
    cv2.putText(frame, status, 
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur_status, 2)

    # Créer une visualisation du masque en couleur
    masque_couleur = cv2.applyColorMap(masque_personne, cv2.COLORMAP_JET)
    
    # Créer overlay avec les zones analysées en cyan
    overlay_zones = frame.copy()
    overlay_zones[masque_bords == 255] = overlay_zones[masque_bords == 255] * 0.6 + np.array([255, 255, 0]) * 0.4

    # Afficher les fenêtres
    cv2.imshow('1. Video + Detection YOLO', frame)
    cv2.imshow('2. Zones Analysees (Jaune = Bords)', overlay_zones)
    cv2.imshow('3. Masque Personne', masque_personne)
    cv2.imshow('4. Arriere-plan Bords Extraits', arriere_plan)
    
    # Appuyer sur 'p' pour pause, 'q' pour quitter
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("\n⏸️  PAUSE - Appuyez sur une touche pour continuer...")
        cv2.waitKey(0)

# ==================== RÉSUMÉ ====================
cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*60}")
print(f"✅ ANALYSE TERMINÉE")
print(f"{'='*60}")
print(f"📊 Frames analysées      : {frame_count}")
print(f"👤 Personne détectée     : {personne_detectee_count}/{frame_count} frames ({(personne_detectee_count/frame_count)*100:.1f}%)")
print(f"⚠️  Changements détectés  : {changements_detectes}")

if changements_detectes == 0:
    print(f"✅ Arrière-plan stable - Validation réussie ✓")
elif changements_detectes < 5:
    print(f"⚠️  Arrière-plan légèrement instable")
else:
    print(f"❌ Arrière-plan instable - Validation échouée ✗")

taux_stabilite = ((frame_count - changements_detectes) / frame_count) * 100
print(f"📈 Taux de stabilité     : {taux_stabilite:.1f}%")
print(f"{'='*60}\n")