import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
)

def analyzer_direction(landmarks, frame_width, frame_height):
    nez = landmarks[1]
    menton = landmarks[152]
    oeil_gauche = landmarks[33]
    oeil_droit = landmarks[263]

    nez_x = int(nez.x * frame_width)
    nez_y = int(nez.y * frame_height)
    menton_x = int(menton.x * frame_width)
    menton_y = int(menton.y * frame_height)
    oeil_gauche_x = int(oeil_gauche.x * frame_width)
    oeil_gauche_y = int(oeil_gauche.y * frame_height)
    oeil_droit_x = int(oeil_droit.x * frame_width)
    oeil_droit_y = int(oeil_droit.y * frame_height)

    centre_yeux_x = (oeil_gauche_x + oeil_droit_x) / 2
    centre_yeux_y = (oeil_gauche_y + oeil_droit_y) / 2

    decalage_x = nez_x - centre_yeux_x
    decalage_y = nez_y - centre_yeux_y

    if decalage_x > 15:
        return "Droite"
    elif decalage_x < -15:
        return "Gauche"
    elif decalage_y > 15:
        return "Bas"
    elif decalage_y < -15:
        return "Haut"
    else:
        return "Centre"
    
video_path = ''
cap = cv2.VideoCapture(video_path)
mouvement_detecte = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        direction = analyzer_direction(landmarks, frame.shape[1], frame.shape[0])
        mouvement_detecte.append(direction)

        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        )

        cv2.putText(frame, f"Direction: {direction}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "aucun visage detecté", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow('analyze mouvements', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Mouvements détectés:")
print(f"   - GAUCHE: {mouvement_detecte.count('Gauche')}")
print(f"   - DROITE: {mouvement_detecte.count('Droite')}")
print(f"   - HAUT: {mouvement_detecte.count('Haut')}")
print(f"   - BAS: {mouvement_detecte.count('Bas')}")
print(f"   - CENTRE: {mouvement_detecte.count('Centre')}")
