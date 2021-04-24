import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import gym
from DeepRL.Async_Actor_Critic.Worker import Worker
from DeepRL.Async_Actor_Critic.Utilities import create_actor_critic_model, accumulate_gradients


class Master:
    def __init__(self, env_name='CartPole-v0', gym_env=True, logging=False, env=None):
        if gym_env:
            self.env = gym.make(env_name)
        else:
            self.env = env

        self.custom_env = not gym_env
        self.env_name = env_name
        self.num_actions = self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape
        self.global_model = create_actor_critic_model(self.observation_shape, self.num_actions, 128)
        self.eps = np.finfo(np.float32).eps.item()
        self.logging = logging

    def train(self, num_processes, num_episodes, gradient_accum_method='weight'):
        num_processes = min(num_processes, mp.cpu_count() - 1)
        if self.logging:
            print(f'Number of processes: {num_processes}')
        response_queue = mp.Queue()
        output_queue = mp.Queue()
        workers = [Worker(env_name=self.env_name, num_actions=self.num_actions, num_episodes=num_episodes,
                          num_observations=self.observation_shape, custom_env=self.custom_env,
                          global_model=self.global_model, eps=self.eps, logging=self.logging,
                          res_queue=response_queue, output_queue=output_queue, worker_index=i) for i in
                   range(num_processes)]

        for i, worker in enumerate(workers):
            if self.logging:
                print(f'Starting worker {i}')
            worker.start()
        optimizer = keras.optimizers.Adam(learning_rate=.01)
        waiting = num_processes
        gradient_sync = num_processes
        gradients = []
        num_episodes = 0
        rewards = []
        ep_rewards = np.zeros(num_processes)

        while waiting != 0:
            worker_id, running_reward, grads = response_queue.get()
            rewards.append(running_reward)
            ep_rewards[worker_id] = running_reward
            if grads is None:
                waiting -= 1
            else:
                # accum_gradient = [(acum + grad) for acum, grad in zip(accum_gradient, grads)]
                gradients.append(grads)
                gradient_sync -= 1
                if gradient_sync == 0:
                    accum_gradient = accumulate_gradients(self.global_model.trainable_variables, gradients,
                                                          num_processes, ep_rewards, gradient_accum_method)
                    optimizer.apply_gradients(
                        zip(accum_gradient, self.global_model.trainable_variables))

                    weights = self.global_model.get_weights()
                    for i in range(num_processes):
                        output_queue.put(weights)

                    gradient_sync = num_processes
                    gradients.clear()
                num_episodes += 1

        [w.join() for w in workers]
        if self.logging:
            print(f"Number of episodes run: {num_episodes}")
        return rewards

    def play(self):
        for i in range(5):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                self.env.render()

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)  # Keras model expects a 2D input

                # Predict action probabilities and critic value from the model state.
                action_probs, critic_val = self.global_model(state)

                # Take a sample from the actor probability distribution.
                action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))

                state, reward, done, _ = self.env.step(action)  # Apply the action to the environment
