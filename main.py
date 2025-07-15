import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow import keras
import argparse
import time
import os


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to the wrist (landmark 0) and scale to unit size.
    """
    landmarks = np.array(landmarks).reshape(21, 3)
    origin = landmarks[0]
    landmarks -= origin
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks


class GestureRecognizer:
    def __init__(self, model_path, encoder_path, camera_id=0, threshold=0.5):
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError("Model (.h5) or encoder (.pkl) file not found.")

        self.model = keras.models.load_model(model_path)

        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.threshold = threshold
        self.cap = cv2.VideoCapture(camera_id)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def run(self):
        prev_time = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    x_min, y_min, x_max, y_max = w, h, 0, 0

                    for lm in hand_landmarks.landmark:
                        x, y, z = lm.x, lm.y, lm.z
                        landmarks.extend([x, y, z])
                        cx, cy = int(x * w), int(y * h)
                        x_min, y_min = min(x_min, cx), min(y_min, cy)
                        x_max, y_max = max(x_max, cx), max(y_max, cy)

                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    # Normalize & reshape for CNN
                    norm_landmarks = normalize_landmarks(landmarks)
                    X_input = norm_landmarks.reshape(1, 9, 7)

                    # Predict
                    probs = self.model.predict(X_input, verbose=0)[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]

                    if confidence >= self.threshold:
                        pred_name = self.label_encoder.inverse_transform([pred_idx])[0]
                    else:
                        pred_name = "Unknown"

                    # Draw bounding box & label
                    cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (255, 255, 0), 2)
                    cv2.putText(frame, f"{pred_name} ({confidence:.2f})", (x_min, y_min-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="exports/cnn_model.h5", help="Path to CNN model (.h5)")
    parser.add_argument("--encoder", type=str, default="exports/cnn_encoder.pkl", help="Path to LabelEncoder (.pkl)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")

    args = parser.parse_args()

    recognizer = GestureRecognizer(
        model_path=args.model,
        encoder_path=args.encoder,
        camera_id=args.camera,
        threshold=args.threshold
    )

    recognizer.run()
