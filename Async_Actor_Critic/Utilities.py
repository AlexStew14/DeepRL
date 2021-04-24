import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_actor_critic_model(num_inputs, num_actions, num_hidden):
    inputs = layers.Input(shape=list(num_inputs))
    common = layers.Dense(num_hidden, activation='relu')(inputs)
    flat = layers.Flatten()(common)
    actor = layers.Dense(num_actions, activation='softmax')(flat)  # Actor layer
    critic = layers.Dense(1)(flat)  # Critic layer

    return keras.Model(inputs=inputs, outputs=[actor, critic])


def accumulate_gradients(trainable_variables, gradient_list, num_processes, rewards_array, method='weight'):
    accum_gradient = [tf.zeros_like(v) for v in trainable_variables]
    if method == 'weight':
        total_rewards = np.sum(rewards_array)
        # Computes weighted average of gradients based on rewards
        for i in range(num_processes):
            # worker_grad = [(tf.convert_to_tensor(t)) for t in gradients[i]]
            accum_gradient = [(acum + ((rewards_array[i] / total_rewards) * grad)) for
                              acum, grad in zip(accum_gradient, gradient_list[i])]
    elif method == 'best':
        best_worker = np.argmax(rewards_array)
        accum_gradient = [(acum + grad) for acum, grad in zip(accum_gradient, gradient_list[best_worker])]

    return accum_gradient
