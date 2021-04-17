import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
from DeepRL.Async_Actor_Critic.Utilities import create_actor_critic_model


class Worker(mp.Process):
    def __init__(self, res_queue, env_name, num_actions, num_observations,
                 global_model, num_episodes, eps, worker_index):
        super().__init__()
        self.res_queue = res_queue
        self.global_model = global_model
        self.local_model = create_actor_critic_model(num_inputs=num_observations,
                                                     num_actions=num_actions, num_hidden=28)
        self.opt = keras.optimizers.Adam(learning_rate=.01)
        self.loss = keras.losses.Huber()
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.env = gym.make(env_name).unwrapped
        self.num_episodes = num_episodes
        self.eps = eps
        self.worker_index = worker_index

    def run(self):
        gamma = .99
        # Structures for storing trajectory.
        a_probs_hist = []
        c_val_history = []
        rewards_history = []
        running_reward = 0

        for i in range(self.num_episodes):
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
                    if done:
                        reward = -1
                    rewards_history.append(reward)
                    episode_reward += reward

                # Update running reward as weighted sum. 5% for new trajectory, 95% for old sum.
                running_reward = .05 * episode_reward + .95 * running_reward
                self.res_queue.put(running_reward)

                # Calculate expected value from rewards.
                # This is done with gamma discounting, with gamma being the discount rate.
                returns = []
                discounted_sum = 0
                for r in rewards_history[::-1]:  # iterate backwards
                    discounted_sum = r + gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize returns
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
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
                    critic_losses.append(self.loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

                # Backpropagation to update the model.
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.local_model.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.global_model.get_weights()))

                # Clear histories since each training step is one trajectory.
                a_probs_hist.clear()
                c_val_history.clear()
                rewards_history.clear()

        # Indicate to master thread that this worker is done training.
        self.res_queue.put(None)
