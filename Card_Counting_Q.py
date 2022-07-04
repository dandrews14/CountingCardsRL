import math
import numpy as np
import gym
import colorama
import time
import sys
import random

# Rules:
#   Dealer Hits on anything less than 17
#   6 Decks played straight through
#   3:2 Blackjack


def encodeState(s1,s2,s3,s4):
  #print(s1,s2,s3,s4)
  if s3:
      s3 = 1
  else:
      s3 = 0
  
  s4 = getS4(s4)

  output = s1
  output *= 12
  output += s2
  output *= 8
  output += s4
  output *= 2
  output += s3
  return output 

def getS4(s4):
    if s4 <= -6:
        s4 = 0
    elif s4 <= -4:
        s4 = 1
    elif s4 <= -3:
        s4 = 2
    elif s4 <= -1:
        s4 = 3
    elif s4 <= 1:
        s4 = 4
    elif s4 <= 3:
        s4 = 5
    elif s4 <= 4:
        s4 = 6
    else:
        s4 = 7
    return s4

class Deck:
    def __init__(self):
        self.cards = [11,2,3,4,5,6,7,8,9,10,10,10,10]*24
        random.shuffle(self.cards)
        self.count = 0
        self.TC = 0

    def shuff(self):
        random.shuffle(self.cards)

    def draw(self):
        try:
            c = self.cards.pop(0)
        except:
            self.cards = [11,2,3,4,5,6,7,8,9,10,10,10,10]*24
            self.shuff()
            c = self.cards.pop(0)
        if 2 <= c <= 6:
            self.count += 1
        elif c == 11 or c == 10:
            self.count -= 1

        self.TC = self.count // (1+(len(self.cards))//52)
        return c


class Game:

    def hit(self, player, deck, dealer, s3):
        player.append(deck.draw())
        if sum(player) > 21:
            if s3 == 1:
                while True:
                    player[player.index(11)] = 1
                    s3 = 1 if 11 in player else 0
                    if sum(player) <= 21:
                        complete = 0
                        reward = 0
                        break
                    if sum(player) > 21 and s3 == 0:
                        complete = 1
                        reward = -1
                        break

            else:
                complete = 1
                reward = -1
        #elif sum(player) == 21:
        #    if sum(dealer) != 21:
        #        reward = 1
        #    else:
        #        reward = 0
        #    complete = 1
        else:
            reward = 0
            complete = 0
        return player, reward, complete

    def stand(self, player, dealer, deck):
        while sum(dealer) < 17:
            dealer.append(deck.draw())
        if sum(dealer) > 21:
            if 11 in dealer:
                dealer[dealer.index(11)] = 1
                while sum(dealer) < 17:
                    dealer.append(deck.draw())
                if sum(dealer) > 21:
                    if sum(player) == 21:
                        reward = 1.5
                    else:
                        reward = 1
                elif sum(dealer) < sum(player):
                    if sum(player) == 21:
                        reward = 1.5
                    else:
                        reward = 1
                elif sum(dealer) == sum(player):
                    reward = 0
                else:
                    reward = -1
            else:
                reward = 1
        elif sum(dealer) < sum(player):
            if sum(player) == 21:
                reward = 1.5
            else:
                reward = 1
        elif sum(dealer) == sum(player):
            reward = 0
        else:
            reward = -1

        complete = 1

        return reward, complete

    def doubledown(self, player, dealer, deck, s3):
        player.append(deck.draw())
        if sum(player) > 21:
            if s3 == 1:
                while True:
                    player[player.index(11)] = 1
                    s3 = 1 if 11 in player else 0
                    if sum(player) <= 21:
                        break
                    if sum(player) > 21 and s3 == 0:
                        break
        while sum(dealer) < 17:
            dealer.append(deck.draw())
        if sum(player) > 21:
            return -2, 1
        if sum(dealer) > 21:
            if 11 in dealer:
                dealer[dealer.index(11)] = 1
                while sum(dealer) < 17:
                    dealer.append(deck.draw())
                if sum(dealer) > 21:
                    if sum(player) == 21:
                        reward = 1.5
                    else:
                        reward = 1
                elif sum(dealer) < sum(player):
                    if sum(player) == 21:
                        reward = 1.5
                    else:
                        reward = 1
                elif sum(dealer) == sum(player):
                    reward = 0
                else:
                    reward = -1
            else:
                reward = 1
        elif sum(dealer) < sum(player):
            if sum(player) == 21:
                reward = 1.5
            else:
                reward = 1
        elif sum(dealer) == sum(player):
            reward = 0
        else:
            reward = -1

        complete = 1
        #print(reward*2)
        return int(reward*2), complete

    def start(self, deck):
        # Deal in Dealer
        dealer = []
        dealer.append(deck.draw())
        dealer.append(deck.draw())

        # Deal in Player
        player = []
        player.append(deck.draw())
        player.append(deck.draw())

        s1 = sum(player)
        s2 = dealer[1]
        s3 = 1 if 11 in player else 0
        s4 = getS4(deck.TC)

        return player, dealer, s1, s2, s3, s4



def Q_learn(gamma, alpha, epsilon, n_episodes, decay, deck):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    """
    max_steps = 500

    game = Game()
    
    Q = np.zeros((33 * 12 * 2 * 8, 8))
    for ep in range(n_episodes):

        # Start game
        s1, s2, s3, s4 = 0,0,0,getS4(deck.TC)
        state = encodeState(s1,s2,s3,s4)


        #player, dealer, s1, s2, s3, s4 = game.start(deck)

        #state = encodeState(s1,s2,s3,s4)
        
        if not ep%10000:
            print(ep, "epsilon: {}".format(epsilon))

        i = 0
        payout = 0
        while i < max_steps:

            #if np.random.uniform(0,1) < epsilon:
            #    action = random.randint(0, 7) # take random action
            #else:
            if s1 == s2 == 0:
                if np.random.uniform(0,1) < epsilon:
                    action = random.randint(0, 7)
                else:
                    action = np.argmax(Q[state])
            elif len(player) == 2:
                if np.random.uniform(0,1) < epsilon:
                    action = random.randint(0, 2)
                else:
                    action = np.argmax(Q[state][:3])
            else:
                if np.random.uniform(0,1) < epsilon:
                    action = random.randint(0, 2)
                else:
                    action = np.argmax(Q[state][:2])

                #action = np.argmax(Q[state])

            # Choose wager before game starts
            if s2 == 0 and s1 == 0:
                player, dealer, s1, s2, s3, s4 = game.start(deck)
                state2 = encodeState(s1,s2,s3,s4)
                reward = 0
                complete = 0
                payout = action + 1

            elif action == 2:
                reward, complete = game.doubledown(player, dealer, deck, s3)
                s1 = sum(player)
                s3 = 1 if 11 in player else 0

            # Player has decided to hit
            elif action == 1:
                player, reward, complete = game.hit(player, deck, dealer, s3)
                s1 = sum(player)

                s3 = 1 if 11 in player else 0
            
            # Player has decided to hold
            else:
                reward, complete = game.stand(player, dealer, deck)

            s4 = getS4(deck.TC)

            state2 = encodeState(s1,s2,s3,s4)

            # Update payout based off initial wager
            if complete:
                reward *= payout

            # Update Q
            try:
                Q[state][action] = Q[state][action] + alpha*(reward + gamma*Q[state2][np.argmax(Q[state2])] - Q[state][action])
            except:
                print(state,action, state2)
                print(s1,s2,s3,s4)
                print(player,dealer)

            if complete:
                epsilon = epsilon*decay
                break
            
            state = state2
            i += 1

    return Q

def play(gamma, alpha, epsilon, n_episodes, decay, iterations):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    iterations: Number of testing iterations
    """
    np.set_printoptions(threshold=sys.maxsize)
    q = Q_learn(gamma, alpha, epsilon, n_episodes, decay, Deck())
    #mini = learnBetting(1.0, 0.1, 1, 800000, 0.999998, Deck(), q)
    #env = gym.make('Blackjack-v0')
    
    deck = Deck()
    game = Game()

    score = 0
    wins = 0
    losses = 0
    draws = 0
    over5 = 0
    under5 = 0
    o5w = 0
    u5w = 0
    winnings = 0
    for i in range(iterations):

        # Start game
        s1, s2, s3, s4 = 0,0,0,getS4(deck.TC)
        s = encodeState(s1,s2,s3,s4)

        payout = 0

        complete = 0
        if not i % 10000:
            print(i, "######################")
        while not complete:
            
            if s1 == s2 == 0:
                a = np.argmax(q[s])
                if not i % 10000:
                    print("START")
                    print(a,deck.TC)
            elif len(player) == 2:
                a = np.argmax(q[s][:3])
                if not i % 10000:
                    print("PLAYING")
                    print(s1,s2,s3,deck.TC)
                    print(a)
            else:
                a = np.argmax(q[s][:2])
                if not i % 10000:
                    print("PLAYING")
                    print(s1,s2,s3,deck.TC)
                    print(a)


            if s2 == 0 and s1 == 0:
                player, dealer, s1, s2, s3, s4 = game.start(deck)
                state2 = encodeState(s1,s2,s3,s4)
                reward = 0
                complete = 0
                payout = a + 1

            elif a == 2:
                if sum(player) == 11:
                    print(a)
                reward, complete = game.doubledown(player, dealer, deck, s3)
                #print(f"{deck.TC}, {s1,s2,s3}")

            elif a == 1:
                if sum(player) == 11:
                    print(a)
                player, reward, complete = game.hit(player, deck, dealer, s3)
                s1 = sum(player)

                s3 = 1 if 11 in player else 0
            else:
                if sum(player) == 11:
                    print(a)
                reward, complete = game.stand(player, dealer, deck)
            # Get new state & reward from environment
            #s1,r,d,_ = env.step(a)
            if complete:

                # Winnings Tracker
                winnings += reward * payout

                # Update Tracker
                if deck.TC >= 4:
                    over5 += 1
                    o5w += reward
                elif deck.TC <= 0:
                    under5 += 1
                    u5w += reward


                # Update Wins
                if reward == 1:
                    wins += 1
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1

            s4 = getS4(deck.TC)

            #s = s1
            s = encodeState(s1,s2,s3,s4)
    print(f"The agents average score was {((wins-losses)/iterations)}, and won {wins} times, lost {losses} times, drawed {draws} times")
    print(f"Over = {o5w/over5}", f"Under = {u5w/under5}")
    print(f"{o5w}, {over5}, {u5w}, {under5}")
    print(f"Earnings: {winnings}")
    return q

Q = play(1.0, 0.1, 1, 2000000, 0.999998, 500000)


#mini = learnBetting(1.0, 0.1, 1, 800000, 0.999998, Deck(), Q)

