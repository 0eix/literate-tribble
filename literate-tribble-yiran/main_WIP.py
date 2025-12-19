import os
import time

import cv2
import numpy as np
import pickle
from lib import game, gui, init_dir, models
from lib.utils import draw_landmarks_on_image
from scipy.signal import argrelextrema

APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


PLAYER1 = game.Player("Player 1")

PLAYER2 = game.Player("Player 2")
PLAYER1.gesture = game.Gesture.ROCK

PLAYER2.gesture = game.Gesture.PAPER
GAME = game.Game(PLAYER1, PLAYER2)
# Load NN model
with open("literate-tribble-yiran/models/model_new.pkl", "rb") as f:
    modelNN = pickle.load(f)

def extract_landmarks(hand):
    coords = np.array([[lm.x, lm.y] for lm in hand])
    return coords

def preprocess_landmark(landmark):
    landmark[:, 0::2] -= np.min(landmark[:, 0::2], axis=1, keepdims=True)
    landmark[:, 1::2] -= np.min(landmark[:, 1::2], axis=1, keepdims=True)
    return landmark

def decode_prediction(prediction):
    gesture = ['paper', 'rock', 'scissors','None']
    if (prediction>.8).any():
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
    else:
        return None


def draw_plot(image, values, color=(0, 255, 0), thickness=2, n_minima=2):
    h, w, _ = image.shape

    # Safety checks
    if values is None or len(values) < 3:
        return image

    values = np.asarray(values, dtype=np.float32)

    # Normalize to image height
    values_norm = (values - values.min()) / (values.ptp() + 1e-6)
    ys = h - (values_norm * (h - 50)).astype(int)  # invert y-axis
    xs = np.linspace(0, w - 1, len(values)).astype(int)

    # Draw plot line
    for i in range(len(values) - 1):
        cv2.line(
            image,
            (xs[i], ys[i]),
            (xs[i + 1], ys[i + 1]),
            color,
            thickness
        )

    # Detect local minima
    minima_idx = argrelextrema(values, np.less)[0]

    # Keep first N minima (or deepest if you want)
    minima_idx = minima_idx[:n_minima]

    # Draw minima points
    for i in minima_idx:
        cv2.circle(image, (xs[i], ys[i]), 6, (0, 0, 255), -1)

    return image
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]
    if fg.shape[2] == 3:
        bg[y:y+h, x:x+w] = fg
    else:  # RGBA
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * fg[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )

def main():
    init_dir(APP_ROOT_DIR)
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    history_CoM = []
    threshold = .025
    PROCESS_EVERY = 1     # MediaPipe
    SIGNAL_EVERY = 1     # extrema detection
    DOWN_THRESHOLD = -0.01   # fast downward movement
    UP_THRESHOLD   =  0.01   # fast upward movement
    MIN_FRAMES     = 6      # debounce
    motion_state = "idle"
    desired_step = "up"
    motion_count = 0
    state_step = 0 
    frame_id = 0
    PREDICT_DELAY_FRAMES = int(2 * fps)  # ~0.5 seconds
    round_start_frame = None
    h, w =  150,150
    gesture_up = cv2.imread("literate-tribble-yiran/assets/up_image.png")
    gesture_down = cv2.imread("literate-tribble-yiran/assets/down_image.png")
    gesture_up = cv2.resize(gesture_up, (w, h))
    gesture_down = cv2.resize(gesture_down, (w, h))
    x_img, y_img = 200, 50  # top-left corner


    last_result = None
    try:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            frame_id += 1
            timestamp = int(frame_id * (1000 / max(fps, 30)))

            if frame_id % PROCESS_EVERY == 0:
                models.process_image(image, timestamp)
                last_result = models.latest_result
            else:
                models.latest_result = last_result
            # models.process_image(image, int((time.time() - start_time) * 1000))

            if not success:
                print("Ignoring empty camera frame.")
                continue  # If loading a video, use 'break' instead of 'continue'.

            
# WIP -------------------------------------------------------------------------------------------------------
            if models.latest_result:
                result_image = draw_landmarks_on_image(image, models.latest_result)
                # result_image = image
                if models.latest_result.hand_landmarks:
                    CoM = np.mean([p.y for p in models.latest_result.hand_landmarks[0]],)
                    history_CoM.append(CoM)
                    if len(history_CoM)>int(fps*1.5):
                        history_CoM = history_CoM[-int(fps*1.5):]
                    if len(history_CoM) >= 3:
                        dy = history_CoM[-1] - history_CoM[-2]
                        if desired_step == "up":
                            if dy > UP_THRESHOLD:
                                motion_count += 1
                                if motion_count >= MIN_FRAMES:
                                    state_step += 1 
                                    desired_step = "down"
                                    motion_count = 0 
                            # else:
                                # motion_count = 0
                        elif desired_step == "down":
                            if dy < DOWN_THRESHOLD:
                                motion_count += 1
                                if motion_count >= MIN_FRAMES:
                                    state_step += 1 
                                    desired_step = "up"
                                    motion_count = 0 
                            # else:
                            #     motion_count = 0
                        gesture_text = state_to_gesture(state_step)
                        if gesture_text:
                            cv2.putText(
                                result_image,
                                gesture_text,
                                (250, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 255, 0),
                                3
                            )
                        if state_step in [0,2,4]:
                            gesture_img = gesture_down
                            # result_image[y_img:y_img+h, x_img:x_img+w] = gesture_img
                            overlay_image(result_image, gesture_img, x_img, y_img)
                        elif state_step in [1,3]:
                            gesture_img = gesture_up
                            # result_image[y_img:y_img+h, x_img:x_img+w] = gesture_img
                            overlay_image(result_image, gesture_img, x_img, y_img)
                        if state_step == 5:
                            GAME.round_on_going = True
                            round_start_frame = frame_id
                            state_step = 0 
                            motion_count = 0 
                            desired_step = "up"
                            history_CoM.clear()
                            print("Round started")
# ------------------------------------------------------------------------------------------------------------------------
                    # WORK BUT SLOW -------------------------------------------------------------------------------------
                    # if frame_id % SIGNAL_EVERY == 0 and len(history_CoM) > 10:
                    #     arr = np.asarray(history_CoM, dtype=np.float32)
                    #     mins = argrelextrema(arr, np.less)[0]
                    #     maxs = argrelextrema(arr, np.greater)[0]                        
                    #     if len(maxs)>=3 and len(mins)>=3:
                    #         if sum(arr[maxs[:3]]-arr[mins[:3]]>threshold) == 3:
                    #             print("Round started")
                    #             # GAME.round_start_time = time.time()
                    #             GAME.round_on_going = True
                    #             history_CoM = []
                    # -------------------------------------------------------------------------------------------
                # if not GAME.round_on_going:
                #     if (
                #         len(models.latest_result.gestures) == 2
                #         and models.latest_result.gestures[0][0].category_name
                #         == "Thumb_Up"
                #         and models.latest_result.gestures[1][0].category_name
                #         == "Thumb_Up"
                #     ):
                #         print("Round started")
                #         GAME.round_start_time = time.time()
                #         GAME.round_on_going = True
                    
            else:
                result_image = image

            # draw interface
            gui.draw_interface(result_image, PLAYER1, PLAYER2)
            if GAME.round_on_going:
                # Input lag to allow players to make their play
                if frame_id - round_start_frame < PREDICT_DELAY_FRAMES:
        # Still waiting → show camera feed normally
                        cv2.putText(
                            result_image,
                            "Hold gesture...",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                        )
                else:
                    if models.latest_result and len(models.latest_result.gestures) == 2:
                        CoM1 = np.mean(
                            [p.x for p in models.latest_result.hand_landmarks[0]],
                        )
                        CoM2 = np.mean(
                            [p.x for p in models.latest_result.hand_landmarks[1]],
                        )                           
                        
                        try:
                            hand_list = models.latest_result.hand_landmarks
                            if CoM1 < CoM2:
                                left_hand = hand_list[0]
                                right_hand = hand_list[1]
                            else:
                                left_hand = hand_list[1]
                                right_hand = hand_list[0]
                            # Convert to array with x,y coordinates
                            left_landmark = extract_landmarks(left_hand).reshape(1, -1)
                            right_landmark = extract_landmarks(right_hand).reshape(1, -1)
                            # Set min (x,y) to (0,0)
                            left_landmark_processed = preprocess_landmark(left_landmark)
                            right_landmark_processed = preprocess_landmark(right_landmark)
                            # Invert x values for right player
                            right_landmark_processed[:, 0::2] -= np.max(right_landmark_processed[:, 0::2], axis=1, keepdims=True)
                            right_landmark_processed = np.abs(right_landmark_processed)
                            # Predict the probabilities of ['paper', 'rock', 'scissors']
                            prediction_left = modelNN.predict(left_landmark_processed,verbose=0)
                            prediction_right = modelNN.predict(right_landmark_processed,verbose=0)
                            # Return the expected strings ["Closed_Fist","Open_Palm","Victory"]
                            gest_left = decode_prediction(prediction_left)
                            gest_right = decode_prediction(prediction_right)
                            
                            PLAYER1.gesture = game.Gesture(gest_left)
                            PLAYER2.gesture = game.Gesture(gest_right)
                            GAME.judge()
                        except ValueError as e:
                            print(f"Invalid gesture: {e}")
                        print("Round over")
                        GAME.round_on_going = False
            # result_image = draw_plot(result_image, history_CoM)
            cv2.imshow("MediaPipe", result_image)
            
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        models.quit()


if __name__ == "__main__":
    main()
