# Rock, Paper, Scissors Game

Enable two human players to compete in a game of Rock, Paper, Scissors.

---

## Hand Landmark Model

The game uses `vision.GestureRecognizer` instead of Holistic, because it is **faster** and provides a smoother user experience.  
> Note: This comes at a small cost in tracking precision.

---

## Start of the Game

The game is initiated by **3 consecutive up & down hand movements**.  
- Only the hand at the **most right-hand corner** is monitored for these movements.  
- Movements are validated if the **y-coordinate of the hand's center of mass** has been moving up/down for at least **X consecutive frames**.  
- Images are displayed to indicate whether the player should move their hand **up** or **down** to continue the sequence.  
- Text (`Rock`, `Paper`, `Scissors`) indicates the **current step in the sequence**.

---

## Analysing Gesture

Once the game has started:  
- Players are given **some time to hold their gesture** before prediction.  
- A **Neural Network (NN)** model is used, built with a custom dataset from group members and classmates.  
- The NN model:  
  - Has a **few dense layers**, allowing for **high accuracy and fast predictions**.  
  - Outputs a **3-element vector**, representing probabilities for each class (`Rock`, `Paper`, `Scissors`).

---

## Determining the Winner

- The NN model predicts the gesture for each player.  
- If a class has **at least 80% probability**, it is assigned to the corresponding player.  
- The game then follows normal **Rock, Paper, Scissors rules** to decide the winner (or tie).  
- Points:  
  - **Winner** → 1 point  
  - **Tie** → 0 points
  - **Loser** → 0 points
