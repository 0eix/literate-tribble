import os
import time

import cv2

from lib import models
from lib.utils import draw_landmarks_on_image

MODEL_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/models"


def main():
    models.init_model_dir(MODEL_DIR)

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
            else:
                result_image = image

            cv2.imshow("MediaPipe", result_image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        models.quit()


if __name__ == "__main__":
    main()
