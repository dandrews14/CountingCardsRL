# CountingCardsRL
Using Reinforcement Learning to Develop a Card Counting Algorithm.

Agent is trained using Q-Learning and attempts to learn a profitable card counting and betting strategy. Best model currently has an average loss of -0.20496 per hand.

Game and models are coded from scratch. Implements the [Omega II](https://www.gamblingsites.org/casino/blackjack/card-counting/omega-2/) Card Counting System.

## Rules:
* Dealer Hits on anything less than 17
* 6 Decks played straight through
* 3:2 Blackjack
* Player bets anywhere from 1-8 units per hand.

## Card_Counting_Q.py

* Agent is trained using 2,000,000 iterations and tested on 500,000 iterations.
* Actions player can take are:
  * Choosing the initial bet (1-8)
  * Standing (0), Hitting (1), or Doubling Down (2) when it is in the opening deal.
  * Standing (0) or Hitting (1) otherwise
* 7,920 possible states