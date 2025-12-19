import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
from lib import game, gui, init_dir, models
from lib.utils import draw_landmarks_on_image
from scipy.signal import argrelextrema

APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== PLAYERS =====================
PLAYER1 = game.Player("Player 1")
PLAYER2 = game.Player("Player 2")

PLAYER1.gesture = game.Gesture.ROCK
PLAYER2.gesture = game.Gesture.PAPER

GAME = game.Game(PLAYER1, PLAYER2)

# ===================== MODEL =====================
with open("literate-tribble-yiran/models/model_new.pkl", "rb") as f:
    modelNN = pickle.load(f)

# ===================== UTILS =====================
def extract_landmarks(hand):
    coords = np.array([[lm.x, lm.y] for lm in hand])
    return coords

def preprocess_landmark(landmark):
    landmark[:, 0::2] -= np.min(landmark[:, 0::2], axis=1, keepdims=True)
    landmark[:, 1::2] -= np.min(landmark[:, 1::2], axis=1, keepdims=True)
    return landmark

def decode_prediction(prediction):
    gesture = ['paper', 'rock', 'scissors', 'None']
    if (prediction > .8).any():
        ind = np.argmax(prediction)
    else:
        ind = 3
    return gesture[ind]

def state_to_gesture(state_step):
    if state_step == 0:
        return "Rock"
    elif state_step in (1, 2):
        return "Paper"
    elif state_step in (3, 4):
        return "Scissors"
    return None

# ===================== NEW : SCORE COLOR =====================
def score_color(player, other_player):
    if player.score > other_player.score:
        return (0, 255, 0)      # Vert
    elif player.score < other_player.score:
        return (0, 0, 255)      # Rouge
    else:
        return (0, 255, 255)    # Jaune

def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]
    if fg.shape[2] == 3:
        bg[y:y+h, x:x+w] = fg
    else:
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * fg[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )

# ===================== MAIN =====================
def main():
    init_dir(APP_ROOT_DIR)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    history_CoM = []
    UP_THRESHOLD = 0.01
    DOWN_THRESHOLD = -0.01
    MIN_FRAMES = 6

    motion_count = 0
    desired_step = "up"
    state_step = 0
    frame_id = 0

    PREDICT_DELAY_FRAMES = int(2 * fps)
    round_start_frame = None

    gesture_up = cv2.resize(
        (255 * plt.imread("literate-tribble-yiran/assets/up_image.png")).astype(np.uint8),
        (100, 100)
    )
    gesture_down = cv2.resize(
        (255 * plt.imread("literate-tribble-yiran/assets/down_image.png")).astype(np.uint8),
        (100, 100)
    )

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            frame_id += 1
            timestamp = int(frame_id * (1000 / fps))

            models.process_image(image, timestamp)

            if models.latest_result:
                result_image = draw_landmarks_on_image(image, models.latest_result)

                if models.latest_result.hand_landmarks:
                    CoM = np.mean([p.y for p in models.latest_result.hand_landmarks[0]])
                    history_CoM.append(CoM)
                    history_CoM = history_CoM[-int(fps * 1.5):]

                    if len(history_CoM) >= 2:
                        dy = history_CoM[-1] - history_CoM[-2]

                        if desired_step == "up" and dy > UP_THRESHOLD:
                            motion_count += 1
                            if motion_count >= MIN_FRAMES:
                                state_step += 1
                                desired_step = "down"
                                motion_count = 0

                        elif desired_step == "down" and dy < DOWN_THRESHOLD:
                            motion_count += 1
                            if motion_count >= MIN_FRAMES:
                                state_step += 1
                                desired_step = "up"
                                motion_count = 0

                    gesture_img = gesture_up if state_step % 2 else gesture_down
                    overlay_image(result_image, gesture_img, 190, 20)

                    gesture_text = state_to_gesture(state_step)
                    if gesture_text:
                        cv2.putText(
                            result_image,
                            gesture_text,
                            (280, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            3
                        )

                    if state_step == 5:
                        GAME.round_on_going = True
                        round_start_frame = frame_id
                        state_step = 0
                        history_CoM.clear()

            else:
                result_image = image

            # ===================== INTERFACE =====================
            gui.draw_interface(result_image, PLAYER1, PLAYER2)

            # ===================== NEW : COLORED SCORE =====================
            color_p1 = score_color(PLAYER1, PLAYER2)
            color_p2 = score_color(PLAYER2, PLAYER1)

            cv2.putText(
                result_image,
                f"Player 1: {PLAYER1.score}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_p1,
                2
            )

            cv2.putText(
                result_image,
                f"Player 2: {PLAYER2.score}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_p2,
                2
            )

            cv2.imshow("MediaPipe", result_image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        models.quit()

# ===================== RUN =====================
if __name__ == "__main__":
    main()
