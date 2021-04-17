import tensorflow as tf
from DeepRL.Async_Actor_Critic.Master import Master
from DeepRL.Async_Actor_Critic.Worker import Worker

# run on cpu
tf.config.set_visible_devices([], 'GPU')
master_process = Master(env_name='CartPole-v0')

master_process.train(num_episodes=200, num_processes=8)

master_process.play()
