import cv2
from ultralytics import YOLO
import torch
import os

# ==================== SOLUTION PYTORCH 2.6+ ====================
# Désactiver complètement la vérification weights_only
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

print('Chargement du modèle YOLO-Face...')

# ==================== CHARGEMENT MODÈLE ====================
model_path = 'yolov8n-face.pt'

if not os.path.exists(model_path):
    print(f"Modèle introuvable : {model_path}")
    exit()

try:
    model = YOLO(model_path)  # ✅ SANS verbose
    print('Modèle chargé\n')
except Exception as e:
    print(f"Erreur : {e}")
    exit()

# ==================== OUVERTURE VIDÉO ====================
video_path = 'IMG_0006.MOV'

if not os.path.exists(video_path):
    print(f"Vidéo introuvable : {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Impossible d'ouvrir : {video_path}")
    exit()

# ==================== INFOS VIDÉO ====================
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Vidéo : {video_path}")
print(f"FPS : {fps:.2f}")
print(f"Frames : {total_frames}")
print(f"Résolution : {width}x{height}\n")
print("Lecture en cours (appuyez sur 'q' pour quitter)...\n")

# ==================== TRAITEMENT ====================
frame_count = 0
faces_detected_total = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Détecter avec YOLO (verbose=False dans predict, pas __init__)
    results = model.predict(frame, verbose=False, conf=0.5, device='cpu')

    faces_in_frame = 0

    for result in results:
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Coordonnées
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                except:
                    # Si tolist() échoue, utiliser numpy
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                
                # Confiance
                try:
                    confidence = float(box.conf[0])
                except:
                    confidence = float(box.conf[0].cpu().numpy())

                if confidence > 0.5:
                    faces_in_frame += 1
                    faces_detected_total += 1
                    
                    # Dessiner bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Afficher confiance
                    cv2.putText(frame, f'Face: {confidence:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

    # ==================== AFFICHAGE INFOS ====================
    # Compteur de frames
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Nombre de visages dans cette frame
    cv2.putText(frame, f"Faces: {faces_in_frame}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Progression
    progression = int((frame_count / total_frames) * 100)
    cv2.putText(frame, f"Progress: {progression}%", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Afficher la frame
    cv2.imshow('Face Detection - YOLO', frame)

    # Quitter avec 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("\nArrêt demandé")
        break

# ==================== NETTOYAGE ====================
cap.release()
cv2.destroyAllWindows()

# ==================== RÉSUMÉ ====================
print(f"\n{'='*50}")
print(f"ANALYSE TERMINÉE")
print(f"{'='*50}")
print(f"Frames analysées : {frame_count}/{total_frames}")
print(f"Visages détectés : {faces_detected_total}")
if frame_count > 0:
    print(f"Moyenne : {faces_detected_total/frame_count:.2f} visages/frame")
    print(f"Durée vidéo : {frame_count/fps:.2f} secondes")
print(f"{'='*50}\n")
