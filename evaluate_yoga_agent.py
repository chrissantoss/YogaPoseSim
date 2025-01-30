import gym
import numpy as np
from stable_baselines3 import PPO
from yoga_env import YogaEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Create the environment and wrap it with a Monitor for logging
env = YogaEnv()
env = Monitor(env)

# Load the trained model
model = PPO.load("yoga_agent")

# Set up TensorBoard logger
log_dir = "./logs/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Run the model in the environment and log to TensorBoard
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close() 