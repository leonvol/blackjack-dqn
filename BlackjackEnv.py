import numpy as np
import random

_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


# rules taken from: https://bicyclecards.com/how-to-play/blackjack/
# enhancement of OpenAI Gym's 'Blackjack-v0'

class BlackjackEnv:
    def __init__(self):
        self.action_space = 5
        self.observation_space = 5
        self.terminal = False
        self.new = True
        self.player_hand = []
        self.dealer_hand = []
        self.bet = 1
        _ = self.reset()

    def draw_card(self):
        """draws random card from cards and returns its value"""
        return random.choice(_deck)

    def calculate_hand(self, hand):
        """calculates maximal value of hand"""
        n_aces = 0
        total = 0
        for card in hand:
            total += card
            if card == 1:
                n_aces += 1
        while n_aces > 0 and total < 21:
            total += 10
            n_aces -= 1
        return total

    def finish_dealer(self):
        """
        draws cards for dealer until game is finished, returns value of his hand
        will not hit on soft 17
        """
        while self.calculate_hand(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
        return self.calculate_hand(self.dealer_hand)

    def make_reward(self, reward=0):
        """
        creates observation which gets returned
        returns: state-list, reward, finished
        """
        player_val = self.calculate_hand(self.player_hand)
        has_ace = self.player_hand.__contains__(1)
        is_pair = len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1]
        return [player_val, has_ace, is_pair, self.new, self.calculate_hand(self.dealer_hand)], reward, self.terminal

    def reset(self):
        self.bet = 1
        self.new = True
        self.terminal = False
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card()]
        return self.make_reward()

    def stick(self):
        self.terminal = True
        self.new = False
        dealer_val = self.finish_dealer()
        player_val = self.calculate_hand(self.player_hand)
        if player_val > 21:
            return self.make_reward(-self.bet)
        if dealer_val > 21:
            return self.make_reward(self.bet)
        if 21 - player_val < 21 - dealer_val:
            return self.make_reward(self.bet)
        return self.make_reward(-self.bet,)

    def hit(self):
        self.new = False
        self.player_hand.append(self.draw_card())
        if self.calculate_hand(self.player_hand) > 21:
            self.terminal = True
            _ = self.finish_dealer()
            return self.make_reward(-self.bet)
        return self.make_reward()

    def split(self):
        if self.new and self.player_hand[0] == self.player_hand[1]:
            self.player_hand = [self.player_hand[0]]
        self.new = False
        self.player_hand = [self.player_hand[0]]
        self.bet *= 2
        return self.make_reward()

    def double(self):
        player_val = self.calculate_hand(self.player_hand)
        if self.new and 9 <= player_val <= 11:
            self.bet *= 2
        self.new = False
        return self.make_reward()

    def surrender(self):
        self.terminal = True
        self.bet *= -0.5
        return self.make_reward(self.bet)

    def step(self, action):
        """
        makes step based on action, 
        returns new state, reward
        """
        assert (0 <= action <= self.action_space)
        self.new = False
        if self.terminal:
            return self.make_reward()

        if action == 0:  # stick, game finished
            return self.stick()

        elif action == 1:  # hit, draw additional card
            return self.hit()

        elif action == 2:  # split hand, take lower
            return self.surrender()

        elif action == 3:  # double bet
            return self.split()

        # surrender, get half of bet back
        return self.double()
