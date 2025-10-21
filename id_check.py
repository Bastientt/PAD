import cv2
from deepface import DeepFace
import numpy as np

video_path = ''
cap = cv2.VideoCapture(video_path)

embedding_reference = None
seuil_similarite = 0.7

frame_count = 0
results = []

print ("analyze en cours")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 10 != 0:
        continue

    try:
        embedding_result = DeepFace.represent(
            frame,
            model_name ='Facenet',
            enforce_detection = True,
            detector_backend = 'opencv'
        )

        embedding_actuel = np.array(embedding_result[0]['embedding'])

        if embedding_reference is None:
            embedding_reference = embedding_actuel
            print(f"reference etablie (frame {frame_count})")
        else:
            similarite = np.dot(embedding_actuel, embedding_reference) / \
                (np.linalg.norm(embedding_actuel) * np.linalg.norm(embedding_reference))
            
            results.append(similarite)

            if similarite < seuil_similarite:
                print(f"Frame {frame_count}: Identité non reconnue (similarité: {similarite:.2f})")
                couleur = (0, 255, 0) if similarite >= seuil_similarite else (0, 0, 255)
                cv2.putText(frame, f"Similarite: {similarite:.2f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, couleur, 2)
                
    except Exception as e:
        print(f"Frame {frame_count}: Erreur - {str(e)}")
        cv2.putText(frame, "Erreur de detection", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
    cv2.imshow('ID Check', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if results:
    print("\nSimilarités enregistrées:")
    print(f"Similarité moyenne: {np.mean(results):.2f}")
    print(f"Similarité min: {np.min(results):.2f}")
    print(f"Similarité max: {np.max(results):.2f}")
    print(f"Frames analysées: {len(results)}")
