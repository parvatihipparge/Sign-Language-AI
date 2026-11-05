import cv2
import mediapipe as mp
import numpy as np
import os

GESTURES = [
    "Hello", "Goodbye", "ThankYou", "Please", "Yes", "No",
    "Maybe", "OK", "Help", "Water", "Food", "Bathroom",
    "Happy", "Sad", "Love", "Sorry", "Today", "Tomorrow",
    "Later", "Now", "What", "Where", "When", "How",
    "One", "Two", "Three", "Four", "Five", "Want",
    "Need", "Give", "Take", "Mother", "Father",
    "Family", "Friend"
]

SAMPLES_PER_GESTURE = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

LANDMARKS_PATH = os.path.join(DATASET_DIR, "landmarks.npy")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.npy")

# Load existing dataset instead of overwriting it
if os.path.exists(LANDMARKS_PATH) and os.path.exists(LABELS_PATH):
    all_landmarks = np.load(LANDMARKS_PATH).tolist()
    all_labels = np.load(LABELS_PATH).tolist()

    if len(all_landmarks) != len(all_labels):
        raise ValueError("Landmarks and labels have different sample counts.")

    if len(all_landmarks) > 0 and np.asarray(all_landmarks).shape[1] != 126:
        raise ValueError("Existing dataset is not in the 126-value format.")

    print("Existing dataset loaded:", len(all_landmarks), "samples")
else:
    all_landmarks = []
    all_labels = []

def save_dataset():
    np.save(
        LANDMARKS_PATH,
        np.asarray(all_landmarks, dtype=np.float32)
    )
    np.save(
        LABELS_PATH,
        np.asarray(all_labels, dtype=np.int64)
    )

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise SystemExit("Error: Could not open webcam.")

print("\nSIGN LANGUAGE DATA COLLECTION")
print("Press S to start each gesture.")
print("Press Q to save progress and quit.")

try:
    for gesture_index, gesture_name in enumerate(GESTURES):

        existing_count = sum(
            1 for label in all_labels if label == gesture_index
        )

        if existing_count >= SAMPLES_PER_GESTURE:
            print(f"Skipping {gesture_name}: already completed.")
            continue

        collected = existing_count
        collecting = False

        print(
            f"\nGesture: {gesture_name} "
            f"({collected}/{SAMPLES_PER_GESTURE})"
        )

        while collected < SAMPLES_PER_GESTURE:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            cv2.putText(
                frame, f"Gesture: {gesture_name}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Samples: {collected}/{SAMPLES_PER_GESTURE}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            status = "COLLECTING" if collecting else "PRESS S TO START"

            cv2.putText(
                frame, status,
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.imshow("Sign Language Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                save_dataset()
                print("\nProgress saved. Collection stopped.")
                raise KeyboardInterrupt

            if key == ord("s") and not collecting:
                collecting = True
                print(f"Started collecting {gesture_name}.")

            if not collecting:
                continue

            if not results.multi_hand_landmarks:
                continue

            hand_data = {
                "Left": None,
                "Right": None
            }

            if results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    hand_label = handedness.classification[0].label

                    if hand_label in hand_data:
                        hand_data[hand_label] = hand_landmarks

            sample = []

            for hand_label in ["Left", "Right"]:
                if hand_data[hand_label] is not None:
                    for landmark in hand_data[hand_label].landmark:
                        sample.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])
                else:
                    sample.extend([0.0] * 63)

            if len(sample) != 126:
                continue

            all_landmarks.append(sample)
            all_labels.append(gesture_index)
            collected += 1

            if collected % 10 == 0:
                save_dataset()

            print(
                f"\r{gesture_name}: "
                f"{collected}/{SAMPLES_PER_GESTURE}",
                end=""
            )

        if collected < SAMPLES_PER_GESTURE:
            break

        save_dataset()
        print(f"\nCompleted {gesture_name}.")

    print("\nCollection session finished.")
    print("Total samples:", len(all_landmarks))

except KeyboardInterrupt:
    print("\nCollection interrupted.")

finally:
    save_dataset()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Dataset progress saved.")