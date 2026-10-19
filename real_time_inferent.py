import os
import time

import cv2
import numpy as np

from face_runtime import create_face_app, normalized_embedding


PROTOTYPES_FILE = "face_prototypes.npz"
CAMERA_INDEX = 0

if not os.path.exists(PROTOTYPES_FILE):
    raise FileNotFoundError(
        f"Không tìm thấy {PROTOTYPES_FILE}; hãy chạy preprocessing.py trước."
    )

with np.load(PROTOTYPES_FILE, allow_pickle=False) as model:
    prototype_labels = model["labels"].astype(str)
    prototypes = model["prototypes"].astype(np.float32)
    cosine_threshold = float(model["cosine_threshold"].item())

app = create_face_app()
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

previous_time = 0.0
print("[INFO] Bắt đầu realtime; nhấn 'q' để thoát.")
print(f"[INFO] Cosine threshold: {cosine_threshold:.2f}")
while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Không đọc được frame từ camera.")
        break

    current_time = time.time()
    fps = 1 / (current_time - previous_time) if previous_time else 0
    previous_time = current_time

    for face in app.get(frame):
        bbox = face.bbox.astype(int)
        embedding = normalized_embedding(face)
        if embedding is None:
            name = "Unknown"
            score = 0.0
            color = (0, 0, 255)
        else:
            similarities = prototypes @ embedding
            best_index = int(np.argmax(similarities))
            score = float(similarities[best_index])
            if score >= cosine_threshold:
                name = prototype_labels[best_index]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

        label = f"{name} ({score:.2f})"
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        text_top = max(0, bbox[1] - 25)
        cv2.rectangle(frame, (bbox[0], text_top), (bbox[0] + len(label) * 11, bbox[1]), color, -1)
        cv2.putText(
            frame,
            label,
            (bbox[0], max(15, bbox[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Face Recognition Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()