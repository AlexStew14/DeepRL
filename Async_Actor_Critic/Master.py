import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import gym
from DeepRL.Async_Actor_Critic.Worker import Worker
from DeepRL.Async_Actor_Critic.Utilities import create_actor_critic_model


class Master:
    def __init__(self, env_name):
        env = gym.make(env_name)
        self.env_name = env_name
        self.num_actions = env.action_space.n
        self.num_observations = env.observation_space.shape[0]
        self.global_model = create_actor_critic_model(self.num_observations, self.num_actions, 128)
        self.eps = np.finfo(np.float32).eps.item()

    def train(self, num_processes, num_episodes):
        num_processes = min(num_processes, mp.cpu_count())
        mp.set_start_method('spawn')
        response_queue = mp.Queue()
        workers = [Worker(env_name=self.env_name, num_actions=self.num_actions, num_episodes=num_episodes,
                          num_observations=self.num_observations,
                          global_model=self.global_model, eps=self.eps,
                          res_queue=response_queue, worker_index=i) for i in range(num_processes)]

        for i, worker in enumerate(workers):
            print(f'Starting worker {i}')
            worker.start()

        moving_average_rewards = []
        while True:
            reward = response_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break

        [w.join() for w in workers]
        plt.plot(moving_average_rewards)
        plt.show()
