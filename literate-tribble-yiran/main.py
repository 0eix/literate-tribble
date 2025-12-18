import os
import time

import cv2
import numpy as np
import pickle
from lib import game, gui, init_dir, models
from lib.utils import draw_landmarks_on_image

APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


PLAYER1 = game.Player("Player 1")

PLAYER2 = game.Player("Player 2")
PLAYER1.gesture = game.Gesture.ROCK

PLAYER2.gesture = game.Gesture.PAPER
GAME = game.Game(PLAYER1, PLAYER2)
# Load NN model
with open("model.pkl", "rb") as f:
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
    ROCK = "Closed_Fist"
    PAPER = "Open_Palm"
    SCISSORS = "Victory"
    gesture = [PAPER,ROCK,SCISSORS,"None"]
    if (prediction>.8).any():
        ind = np.argmax(prediction)
    else:
        ind = 3
    return gesture[ind]



def main():
    init_dir(APP_ROOT_DIR)

    start_time = time.time()

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)

            if not success:
                print("Ignoring empty camera frame.")
                continue  # If loading a video, use 'break' instead of 'continue'.

            models.process_image(image, int((time.time() - start_time) * 1000))

            if models.latest_result:
                result_image = draw_landmarks_on_image(image, models.latest_result)                

                if not GAME.round_on_going:
                    if (
                        len(models.latest_result.gestures) == 2
                        and models.latest_result.gestures[0][0].category_name
                        == "Thumb_Up"
                        and models.latest_result.gestures[1][0].category_name
                        == "Thumb_Up"
                    ):
                        print("Round started")
                        GAME.round_start_time = time.time()
                        GAME.round_on_going = True
            else:
                result_image = image
                temp = []

            # draw interface
            gui.draw_interface(result_image, PLAYER1, PLAYER2)
            if GAME.round_on_going:
                countdown = 4.0 - (time.time() - GAME.round_start_time)

                if countdown <= 4:
                    cv2.putText(
                        result_image,
                        f"Ready in {int(countdown)}s",
                        (result_image.shape[1] // 2 - 250, result_image.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        2,
                    )

                if countdown <= 0.5:
                    # use the latest result for players
                    if models.latest_result:
                        if len(models.latest_result.gestures) == 2:
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
                                    # gest_left = game.Gesture(
                                    #     models.latest_result.gestures[0][
                                    #         0
                                    #     ].category_name
                                    # )
                                    # gest_right = game.Gesture(
                                    #     models.latest_result.gestures[1][
                                    #         0
                                    #     ].category_name
                                    # )
                                else:
                                    left_hand = hand_list[1]
                                    right_hand = hand_list[0]
                                    
                                    # gest_left = game.Gesture(
                                    #     models.latest_result.gestures[1][
                                    #         0
                                    #     ].category_name
                                    # )
                                    # gest_right = game.Gesture(
                                    #     models.latest_result.gestures[0][
                                    #         0
                                    #     ].category_name
                                    # )
                                # Convert to array with x,y coordinates
                                left_landmark = extract_landmarks(left_hand).reshape(1, -1)
                                right_landmark = extract_landmarks(right_hand).reshape(1, -1)
                                # Set min (x,y) to (0,0)
                                left_landmark_processed = preprocess_landmark(left_landmark)
                                right_landmark_processed = preprocess_landmark(right_landmark)
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
            cv2.imshow("MediaPipe", result_image)
            
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        models.quit()


if __name__ == "__main__":
    main()
