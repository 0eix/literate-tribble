import cv2

from . import assets
from .game import Player

ICON_SIZE = 64


def draw_interface(image, player1: Player, player2: Player):
    # Player 1
    icon_person = cv2.resize(assets.get_asset("person"), (ICON_SIZE, ICON_SIZE))
    image[10 : 10 + ICON_SIZE, 10 : 10 + ICON_SIZE][icon_person == 0] = icon_person[
        icon_person == 0
    ]

    image = cv2.putText(
        image,
        player1.name,
        (10, 10 + ICON_SIZE + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    image = cv2.putText(
        image,
        f"{player1.score}",
        (10 + ICON_SIZE + 20, 10 + ICON_SIZE + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (0, 0, 255),
        2,
    )

    if player1.gesture:
        gest_icon_1 = cv2.resize(
            assets.get_asset(assets.asset_map(player1.gesture)),
            (int(ICON_SIZE * 0.5), int(ICON_SIZE * 0.5)),
        )
        image[
            10 + ICON_SIZE + 30 : 10 + ICON_SIZE + 30 + int(ICON_SIZE * 0.5),
            10 : 10 + int(ICON_SIZE * 0.5),
        ][gest_icon_1 == 0] = gest_icon_1[gest_icon_1 == 0]

    # Player 2
    icon_person = cv2.resize(assets.get_asset("person"), (ICON_SIZE, ICON_SIZE))
    image[10 : 10 + ICON_SIZE, -(ICON_SIZE + 10) : -10][icon_person == 0] = icon_person[
        icon_person == 0
    ]

    image = cv2.putText(
        image,
        player2.name,
        (image.shape[1] - ICON_SIZE - 10, 10 + ICON_SIZE + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    image = cv2.putText(
        image,
        f"{player2.score}",
        (
            image.shape[1] - ICON_SIZE - 20 - 40 * len(f"{player2.score}"),
            10 + ICON_SIZE + 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        2,
    )

    if player2.gesture:
        gest_icon_2 = cv2.resize(
            assets.get_asset(assets.asset_map(player2.gesture)),
            (int(ICON_SIZE * 0.5), int(ICON_SIZE * 0.5)),
        )
        image[
            10 + ICON_SIZE + 30 : 10 + ICON_SIZE + 30 + int(ICON_SIZE * 0.5),
            -(int(ICON_SIZE * 0.5) + 10) : -10,
        ][gest_icon_2 == 0] = gest_icon_2[gest_icon_2 == 0]
