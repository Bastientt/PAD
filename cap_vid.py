import cv2

video_path = ''

cap=cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
print(f"Video Opened: {video_path}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Total Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo")
        break
    frame_count += 1
    cv2.putText(frame, f"Frame: {frame_count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Test Video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f'{frame_count} frames totales')