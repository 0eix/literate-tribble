import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_DIR: str

latest_result: vision.GestureRecognizerResult | None = None  # pyright: ignore
is_processing: bool = False

_recognizer: vision.GestureRecognizer | None = None  # pyright: ignore


def result_callback(
    result: vision.GestureRecognizerResult,  # pyright: ignore
    output_image: mp.Image,
    timestamp_ms: int,
):
    global latest_result, is_processing
    latest_result = result
    is_processing = False


def init_model_dir(model_dir: str):
    global MODEL_DIR
    MODEL_DIR = model_dir


def get_gesture_recognizer() -> vision.GestureRecognizer:  # pyright: ignore
    global _recognizer

    if _recognizer is None:
        base_options = python.BaseOptions(
            model_asset_path=f"{MODEL_DIR}/gesture_recognizer.task"
        )
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=result_callback,
        )
        _recognizer = vision.GestureRecognizer.create_from_options(options)

    return _recognizer


def process_image(image, frame_id: int):
    global is_processing

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    )

    if not is_processing:
        is_processing = True
        get_gesture_recognizer().recognize_async(mp_image, frame_id)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def quit():
    global _recognizer
    if _recognizer is not None:
        _recognizer.close()
        _recognizer = None


__all__ = [
    "init_model_dir",
    "get_gesture_recognizer",
    "latest_result",
    "is_processing",
    "quit",
]
