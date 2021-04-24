import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
from DeepRL.Async_Actor_Critic.Utilities import create_actor_critic_model
from DeepRL.Environments.Blackjack import blackjack_env


class Worker(mp.Process):
    def __init__(self, res_queue, output_queue, env_name, num_actions, num_observations,
                 global_model, num_episodes, eps, worker_index, logging, custom_env):
        super().__init__()
        self.res_queue = res_queue
        self.output_queue = output_queue
        self.global_model = global_model
        self.local_model = create_actor_critic_model(num_inputs=num_observations,
                                                     num_actions=num_actions, num_hidden=128)
        self.opt = keras.optimizers.Adam(learning_rate=.01)
        self.loss = keras.losses.Huber()
        self.num_actions = num_actions
        self.num_observations = num_observations
        if not custom_env:
            self.env = gym.make(env_name)
        else:
            self.env = blackjack_env.BlackjackEnv()
        self.num_episodes = num_episodes
        self.eps = eps
        self.worker_index = worker_index
        self.logging = logging

    def run(self):
        if self.logging:
            print(f"Worker: {self.worker_index} entered run.")
        gamma = .99
        running_reward = 0

        for i in range(self.num_episodes):
            # Structures for storing trajectory.
            a_probs_hist = []
            c_val_history = []
            rewards_history = []
            if self.logging and i % 25 == 0:
                print(f"Worker: {self.worker_index} on episode: {i}.")
                print(f'worker: {self.worker_index}, running reward: {running_reward}')
            state = self.env.reset()
            episode_reward = 0
            done = False
            with tf.GradientTape() as tape:
                while not done:
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)  # Keras model expects a 2D input

                    # Predict action probabilities and critic value from the model state.
                    action_probs, critic_val = self.local_model(state)
                    c_val_history.append(critic_val[0, 0])  # Append the one critic value predicted.

                    # Take a sample from the actor probability distribution.
                    action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
                    # Add the log probability for the specific action taken by the actor.
                    a_probs_hist.append(tf.math.log(action_probs[0, action]))

                    state, reward, done, _ = self.env.step(action)  # Apply the action to the environment

                    rewards_history.append(reward)
                    episode_reward += reward

                # Update running reward as weighted sum. 5% for new trajectory, 95% for old sum.
                running_reward = .05 * episode_reward + .95 * running_reward
                # running_reward = episode_reward

                # Calculate expected value from rewards.
                # This is done with gamma discounting, with gamma being the discount rate.
                returns = np.zeros(len(rewards_history))
                discounted_sum = 0
                count = len(rewards_history) - 1
                for r in rewards_history[::-1]:  # iterate backwards
                    discounted_sum = r + gamma * discounted_sum
                    returns[count] = discounted_sum
                    count -= 1

                # Normalize returns
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)

                a_probs_hist = tf.convert_to_tensor(a_probs_hist)
                c_val_history = tf.convert_to_tensor(c_val_history)
                critic_losses = self.loss(c_val_history, returns)
                actor_losses = (-a_probs_hist * (returns - c_val_history))

                # Backpropagation to update the model.
                loss_value = sum(actor_losses) + (critic_losses * len(returns))
                grads = tape.gradient(loss_value, self.local_model.trainable_variables)
                self.res_queue.put((self.worker_index, running_reward, grads))
                # self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                # Clear histories since each training step is one trajectory.
                weights = self.output_queue.get()
                self.local_model.set_weights(weights)

        # Indicate to master thread that this worker is done training.
        if self.logging:
            print(f"Worker: {self.worker_index} exited run")
        self.res_queue.put((None, None, None))
