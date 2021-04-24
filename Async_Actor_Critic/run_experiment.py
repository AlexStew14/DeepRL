import tensorflow as tf
from DeepRL.Async_Actor_Critic.Master import Master
from DeepRL.Async_Actor_Critic.Worker import Worker
from DeepRL.Environments.Blackjack.blackjack_env import BlackjackEnv
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import time

# run on cpu
tf.config.set_visible_devices([], 'GPU')
env = BlackjackEnv()
master_process = Master(env=env, gym_env=False, logging=True)

num_episodes = 5_000
num_processes = 8
start = time.time()
rewards = master_process.train(num_episodes=num_episodes, num_processes=num_processes, gradient_accum_method='weight')
end = time.time()
total_time = round(end - start, 2)
master_process.play()
plt.plot(rewards)
plt.title(f'Processes: {num_processes}, episodes: {num_episodes}, time taken: {total_time} seconds')
# plt.savefig(f'Figures/{num_processes},{num_episodes},{total_time}.png')
plt.show()
