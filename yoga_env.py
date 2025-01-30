import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random

class YogaEnv(gym.Env):
    def __init__(self):
        super(YogaEnv, self).__init__()
        self.physicsClient = p.connect(p.DIRECT)  # Use p.GUI for visualization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        self.humanoidId = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 1], useFixedBase=False)
        p.setGravity(0, 0, -9.81)
        self.timeStep = 1.0 / 240.0
        p.setTimeStep(self.timeStep)

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Example action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)  # Adjusted shape

    def step(self, action):
        # Apply action to the robot
        # Assuming action is a single integer, use it directly
        # Here, you might want to map the action to a specific joint control
        # For example, if action is 0, move joint to position 0, if 1, move to position 1
        target_position = 0.0 if action == 0 else 1.0
        p.setJointMotorControl2(self.humanoidId, 0, p.POSITION_CONTROL, targetPosition=target_position)

        p.stepSimulation()
        time.sleep(self.timeStep)

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(observation)

        # Check if episode is done
        done = self._check_done(observation)

        return observation, reward, done, {}

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        # Randomize initial position
        initial_position = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 1]
        self.humanoidId = p.loadURDF("humanoid/humanoid.urdf", initial_position, useFixedBase=False)
        p.setGravity(0, 0, -9.81)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_observation(self):
        # Example: get joint states
        joint_states = p.getJointStates(self.humanoidId, range(p.getNumJoints(self.humanoidId)))
        joint_positions = [state[0] for state in joint_states]
        # Ensure the observation matches the defined shape
        return np.array(joint_positions[:15], dtype=np.float32)  # Adjusted to match the shape

    def _calculate_reward(self, observation):
        # Define target joint positions for the "Downward Dog" pose
        # These values are hypothetical and should be adjusted based on your robot's joint configuration
        target_pose = np.array([
            0.0,  # Head/neck angle
            -1.0, # Shoulder angle (arms straight, pointing down)
            0.5,  # Elbow angle (slightly bent)
            -1.5, # Hip angle (legs straight, pointing up)
            0.0,  # Knee angle (straight)
            0.5,  # Ankle angle (slightly bent)
            0.0,  # Additional joints as needed
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ])

        # Calculate the difference between current and target joint positions
        pose_error = np.linalg.norm(observation - target_pose)

        # Reward for minimizing pose error
        pose_reward = -pose_error

        # Additional reward for staying upright
        base_position, _ = p.getBasePositionAndOrientation(self.humanoidId)
        upright_reward = base_position[2]

        return pose_reward + upright_reward

    def _check_done(self, observation):
        # Example: check if robot has fallen
        base_position, _ = p.getBasePositionAndOrientation(self.humanoidId)
        return base_position[2] < 0.5  # Consider done if the robot falls

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect() 