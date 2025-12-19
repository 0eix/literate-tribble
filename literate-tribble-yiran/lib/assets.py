import os

import cv2
import numpy as np
from cv2.typing import MatLike

ASSSETS_DIR: str


def init_assets_dir(assets_dir):
    global ASSSETS_DIR
    ASSSETS_DIR = assets_dir


def get_asset(icon_name: str) -> MatLike:
    filename = f"{ASSSETS_DIR}/{icon_name}.png"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Asset not found: {filename}")
    icon = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    a = np.bitwise_not(icon[..., 3])
    icon = cv2.add(cv2.merge([a, a, a, a]), icon)
    icon = cv2.cvtColor(icon, cv2.COLOR_RGBA2RGB)

    return icon


# for debug only
from .game import Gesture


def asset_map(gest: Gesture):
    return {Gesture.ROCK: "rock", Gesture.PAPER: "paper", Gesture.SCISSORS: "scissors"}[
        gest
    ]


__all__ = ["init_assets_dir", "get_asset", "asset_map"]
