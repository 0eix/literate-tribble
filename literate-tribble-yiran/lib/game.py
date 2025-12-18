from enum import Enum


class Gesture(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.gesture: Gesture | None = None


class Game:
    def __init__(self, player1: Player, player2: Player):
        self.player_a = player1
        self.player_b = player2

        self.round_start_time: float = 0.0
        self.round_on_going: bool = False

    def judge(self):
        if self.player_a.gesture == self.player_b.gesture:
            self.player_a.score += 0
            self.player_b.score += 0
        elif (
            self.player_a.gesture == Gesture.ROCK
            and self.player_b.gesture == Gesture.SCISSORS
        ):
            self.player_a.score += 1
        elif (
            self.player_a.gesture == Gesture.PAPER
            and self.player_b.gesture == Gesture.ROCK
        ):
            self.player_a.score += 1
        elif (
            self.player_a.gesture == Gesture.SCISSORS
            and self.player_b.gesture == Gesture.PAPER
        ):
            self.player_a.score += 1
        else:
            self.player_b.score += 1
