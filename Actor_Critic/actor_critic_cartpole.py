# Actor Critic technique adapted from https://keras.io/examples/rl/actor_critic_cartpole/

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Configuration Parameters
seed = 42
gamma = .99
max_steps_per_episode = 10_000
number_of_episodes = 400
env = gym.make('CartPole-v0')
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # This finds the smallest number such that 0 + eps > 0.

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation='relu')(inputs)
actor = layers.Dense(num_actions, activation='softmax')(common)  # Actor layer
critic = layers.Dense(1)(common)  # Critic layer

model = keras.Model(inputs=inputs, outputs=[actor, critic])


def train_model_complete_trajectory(model, env, render_every_episodes=-1):
    optimizer = keras.optimizers.Adam(learning_rate=.01)
    huber_loss = keras.losses.Huber()
    # Structures for storing trajectory.
    a_probs_hist = []
    c_val_history = []
    rewards_history = []
    running_reward = 0
    # Stored for plotting
    running_reward_history = []
    episode_count = 0

    for i in range(number_of_episodes):
        state = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for _ in range(max_steps_per_episode):
                if render_every_episodes > 0 and episode_count % render_every_episodes == 0:
                    env.render()

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)  # Keras model expects a 2D input

                # Predict action probabilities and critic value from the model state.
                action_probs, critic_val = model(state)
                c_val_history.append(critic_val[0, 0])  # Append the one critic value predicted.

                # Take a sample from the actor probability distribution.
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                # Add the log probability for the specific action taken by the actor.
                a_probs_hist.append(tf.math.log(action_probs[0, action]))

                state, reward, done, _ = env.step(action)  # Apply the action to the environment
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break  # This training generates one complete trajectory per episode.

            # Update running reward as weighted sum. 5% for new trajectory, 95% for old sum.
            running_reward = .05 * episode_reward + .95 * running_reward
            running_reward_history.append(running_reward)

            # Calculate expected value from rewards.
            # This is done with gamma discounting, with gamma being the discount rate.
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:  # iterate backwards
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize returns
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            history = zip(a_probs_hist, c_val_history, returns)
            actor_losses = []
            critic_losses = []
            # Iterate over trajectory and compute loss values for every action.
            for log_prob, value, ret in history:
                # This is the difference in discounted reward of action and the predicted value by the critic.
                diff = ret - value
                # Since optimizing the Actor is a gradient ascent problem, the loss is the negative log probability
                # of the action times the difference between the critic expectation and actual value.
                actor_losses.append(-log_prob * diff)

                # The critic loss is the huber loss between the actual discounted value of the action
                # and the predicted value.
                critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

            # Backpropagation to update the model.
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear histories since each training step is one trajectory.
            a_probs_hist.clear()
            c_val_history.clear()
            rewards_history.clear()

        # Logging
        episode_count += 1
        if episode_count % 10 == 0:
            print(f'running reward: {running_reward} at episode {episode_count}')

    return running_reward_history


reward_history = train_model_complete_trajectory(model, env)

plt.plot(list(range(number_of_episodes)), reward_history)
plt.show()
