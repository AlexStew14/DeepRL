import numpy as np
from enum import Enum


class CardValEnum(Enum):
    NONE = 0,
    ACE = 1,
    TWO = 2,
    THREE = 3,
    FOUR = 4,
    FIVE = 5,
    SIX = 6,
    SEVEN = 7,
    EIGHT = 8,
    NINE = 9,
    TEN = 10,
    JACK = 11,
    QUEEN = 12,
    KING = 13


class CardSuits(Enum):
    NONE = 0,
    SPADES = 1,
    HEARTS = 2,
    CLUBS = 3,
    DIAMONDS = 4


def create_deck():
    card_vals = list(range(1, 14, 1))
    card_suits = list(range(1, 5, 1))
    return np.transpose([np.tile(card_vals, len(card_suits)), np.repeat(card_suits, len(card_vals))])


def shuffle_deck(deck):
    np.random.shuffle(deck)


def get_cards_from_deck(deck, indices):
    cards = np.take(deck, indices, 0)
    deck[indices] = 0
    return cards


def sum_of_cards(card_vals):
    '''
    Returns the sum of cards passed.
    :param card_vals: numpy array of card values (excluding suits).
    :return: tuple of sum of cards with face cards as 10 and ace as 11 (first element) and 1 (second element)
    '''
    c = np.copy(card_vals)
    c[(c == 11) | (c == 12) | (c == 13)] = 10
    ace_one = np.sum(c)
    c[c == 1] = 11
    return np.sum(c), ace_one
