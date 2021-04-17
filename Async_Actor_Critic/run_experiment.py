from DeepRL.Async_Actor_Critic.Master import Master
from DeepRL.Async_Actor_Critic.Worker import Worker

master_process = Master(env_name='CartPole-v0')

master_process.train(num_episodes=100, num_processes=1)
