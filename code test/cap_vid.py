import cv2
import os

video_path = 'IMG_0006.MOV'

if not os.path.exists(video_path):
    print(f"Fichier introuvable : {video_path}")
    print(f"Chemin actuel : {os.getcwd()}")
    exit()

cap = None
for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
    print(f"Test backend {backend}...")
    cap = cv2.VideoCapture(video_path, backend)
    if cap.isOpened():
        print(f"Vidéo ouverte avec backend {backend}")
        break
    cap.release()

if cap is None or not cap.isOpened():
    print("Impossible d'ouvrir la vidéo avec tous les backends")
    print("Convertissez en MP4 avec : ffmpeg -i IMG_0006.MOV output.mp4")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nVidéo : {video_path}")
print(f"FPS : {fps}")
print(f"Total Frames : {total_frames}")
print(f"Résolution : {width}x{height}")
print("\nLecture en cours (appuyez sur 'q' pour quitter)...\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print(f"\nFin de la vidéo ({frame_count} frames lues)")
        break
    
    frame_count += 1
    
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    progression = int((frame_count / total_frames) * 100)
    cv2.putText(frame, f"Progression: {progression}%", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow('Test Video', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        print(f"\nArrêt manuel à la frame {frame_count}")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nRésumé :")
print(f"   - Frames totales lues : {frame_count}")
print(f"   - Durée théorique : {frame_count/fps:.2f} secondes")
