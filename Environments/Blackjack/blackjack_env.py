import card_utils
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

        high = np.zeros((2, 14, 2), dtype=np.int)
        high[:, :, 0] = 13
        high[:, :, 1] = 4
        self.observation_space = gym.spaces.Box(low=np.zeros((2, 14, 2), dtype=np.int), high=high, dtype=np.int)
        self.done = None
        self.action_space = gym.spaces.Discrete(2)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        blackjack = False
        reward = 0
        # Actor decides to hit
        if action == 1:
            self.player_hand[self.player_index, :] = card_utils.get_cards_from_deck(self.deck, self.deck_index)
            self.player_index += 1
            self.deck_index += 1

            # Aces can be either 1 or 11
            p_sum_large, p_sum_small = card_utils.sum_of_cards(self.player_hand[:self.player_index, 0])
            if p_sum_small > 21:
                self.done = True
                reward = -1
            elif p_sum_large == 21 or p_sum_small == 21:
                self.done = True
                blackjack = True

        # Actor decides to stay or hits blackjack
        if action == 0 or blackjack:
            d_sum_large, d_sum_small = card_utils.sum_of_cards(self.dealer_hand[:self.dealer_index, 0])
            while dealer_sum < 17:
        # TODO

    def reset(self):
        self.done = False
        self.deck = card_utils.create_deck()
        card_utils.shuffle_deck(self.deck)
        top_cards = card_utils.get_cards_from_deck(self.deck, list(range(4)))
        self.deck_index = 4
        self.player_hand = np.array([top_cards[0], top_cards[2]])
        self.player_index = 2
        # The dealer's first card is hidden to the player
        self.dealer_hand = np.array([np.array([0, 0]), top_cards[1]])
        self.dealer_index = 2
        padded_state = np.zeros((2, 14, 2), dtype=np.int)
        state = np.array([self.dealer_hand, self.player_hand])
        padded_state[:, :state.shape[1], :] = state
        return padded_state

    def render(self, mode='human'):
        pass
