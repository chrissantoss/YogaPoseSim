import gym
from stable_baselines3 import PPO
from yoga_env import YogaEnv

# Create the environment
env = YogaEnv()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent and visualize the learning process
try:
    for _ in range(10000):  # Total timesteps
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()  # Render the environment to visualize the learning
except KeyboardInterrupt:
    pass

# Save the model
model.save("yoga_agent")

# Close the environment
env.close() 