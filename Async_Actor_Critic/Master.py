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
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.num_actions = self.env.action_space.n
        self.num_observations = self.env.observation_space.shape[0]
        self.global_model = create_actor_critic_model(self.num_observations, self.num_actions, 128)
        self.eps = np.finfo(np.float32).eps.item()

    def train(self, num_processes, num_episodes):
        num_processes = min(num_processes, mp.cpu_count() - 1)
        print(f'Number of processes: {num_processes}')
        response_queue = mp.Queue()
        output_queue = mp.Queue()
        workers = [Worker(env_name=self.env_name, num_actions=self.num_actions, num_episodes=num_episodes,
                          num_observations=self.num_observations,
                          global_model=self.global_model, eps=self.eps,
                          res_queue=response_queue, output_queue=output_queue, worker_index=i) for i in
                   range(num_processes)]

        for i, worker in enumerate(workers):
            print(f'Starting worker {i}')
            worker.start()
        optimizer = keras.optimizers.Adam(learning_rate=.01)
        waiting = num_processes
        num_episodes = 0
        while waiting != 0:
            grads = response_queue.get()
            if grads is None:
                waiting -= 1
            else:
                optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
                num_episodes += 1
                weights = self.global_model.get_weights()
                output_queue.put(weights)

        [w.join() for w in workers]
        print(f"Number of episodes run: {num_episodes}")

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
