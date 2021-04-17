import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_actor_critic_model(num_inputs, num_actions, num_hidden):
    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation='relu')(inputs)
    actor = layers.Dense(num_actions, activation='softmax')(common)  # Actor layer
    critic = layers.Dense(1)(common)  # Critic layer

    return keras.Model(inputs=inputs, outputs=[actor, critic])
