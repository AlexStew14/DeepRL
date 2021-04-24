from DeepRL.Environments.Blackjack import card_utils
import gym
import numpy as np


class BlackjackEnv(gym.Env):
    """
    Author: Alex Stewart
    This environment models the popular card game Blackjack https://en.wikipedia.org/wiki/Blackjack.
    Rule considerations:
      Dealer must hit until at sum of at least 17, and must stay if at sum of at least 17.
      If both the player and dealer bust, the player loses.
      If both the player and dealer hit blackjack, the hand is a push.
      Ace is used as a soft 11, which becomes 1 if the player would bust.
    The only ways for the player to win are:
      Player hits blackjack and dealer does not.
      Dealer busts and player does not.
      Neither dealer or player busts and player has a higher sum of cards.
    Observation space:
      Array of shape (2, 14, 2). Each player has a hand with a max number of 14 cards each. Each card has a value id
      and suit id from card_utils.py. Any cards not in play in the observation space have 0's for all values.
    Action space:
      Binary choice. 0 for stay, 1 for hit.
    """

    def __init__(self):
        self.deck = None
        self.player_hand = None
        self.dealer_hand = None
        self.deck_index = None
        self.player_index = None
        self.dealer_index = None
        self.p_sum_large = None
        self.p_sum_small = None

        high = np.zeros((2, 14, 2), dtype=np.int)
        high[:, :, 0] = 13
        high[:, :, 1] = 4
        self.observation_space = gym.spaces.Box(low=np.zeros((2, 14, 2), dtype=np.int), high=high, dtype=np.int)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        done = False
        reward = 0
        # Actor decides to hit
        if action == 1:
            self.player_hand[self.player_index, :] = card_utils.get_cards_from_deck(self.deck, self.deck_index)
            self.player_index += 1
            self.deck_index += 1

            # Aces can be either 1 or 11
            self.p_sum_large, self.p_sum_small = card_utils.sum_of_cards(self.player_hand[:self.player_index, 0])
            if self.p_sum_small > 21:
                done = True
                reward = -1

        # Actor decides to stay
        if action == 0:
            done = True
            d_sum_large, d_sum_small = card_utils.sum_of_cards(self.dealer_hand[:self.dealer_index, 0])
            while d_sum_large < 17:
                self.dealer_hand[self.dealer_index, :] = card_utils.get_cards_from_deck(self.deck, self.deck_index)
                self.dealer_index += 1
                self.deck_index += 1
                d_sum_large, d_sum_small = card_utils.sum_of_cards(self.dealer_hand[:self.dealer_index, 0])

            d_eff_sum = 0
            if d_sum_small > 21:
                reward = 1
            elif d_sum_large > 21:
                d_eff_sum = d_sum_small
            else:
                d_eff_sum = d_sum_large

            if self.p_sum_small is None:
                self.p_sum_large, self.p_sum_small = card_utils.sum_of_cards(self.player_hand[:self.player_index, 0])

            p_eff_sum = 0
            if reward == 0:
                if self.p_sum_large > 21:
                    p_eff_sum = self.p_sum_small
                else:
                    p_eff_sum = self.p_sum_large

                if p_eff_sum > d_eff_sum:
                    reward = 1
                else:
                    reward = -1

        dealer_hand = np.copy(self.dealer_hand)
        dealer_hand[0, :] = 0
        return np.array([dealer_hand, self.player_hand]), reward, done, {}

    def reset(self):
        self.deck = card_utils.create_deck()
        card_utils.shuffle_deck(self.deck)
        top_cards = card_utils.get_cards_from_deck(self.deck, list(range(4)))
        self.deck_index = 4
        self.player_hand = np.zeros((14, 2), dtype=np.int)
        self.player_hand[:2, :] = np.array([top_cards[0], top_cards[2]])
        self.player_index = 2

        self.dealer_hand = np.zeros((14, 2), dtype=np.int)
        self.dealer_hand[:2, :] = np.array([top_cards[1], top_cards[3]])
        self.dealer_index = 2
        # The dealer's first card is hidden to the player
        dealer_hand = np.copy(self.dealer_hand)
        dealer_hand[0, :] = 0
        return np.array([dealer_hand, self.player_hand])

    def render(self, mode='human'):
        print('\n\n STATE\n')
        dealer_hand = np.copy(self.dealer_hand)
        dealer_hand[0, :] = 0
        print(np.array([dealer_hand, self.player_hand]))
        print('\n\n')
