import gym
from stable_baselines3 import PPO
from yoga_env import YogaEnv

# Create the environment
env = YogaEnv()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("yoga_agent")

# Close the environment
env.close() 