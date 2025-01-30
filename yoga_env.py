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
        # Example: action space for joint control
        self.action_space = spaces.Box(low=-1, high=1, shape=(17,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)

    def step(self, action):
        # Apply action to the robot
        # Example: set joint motor control
        for i in range(len(action)):
            p.setJointMotorControl2(self.humanoidId, i, p.POSITION_CONTROL, targetPosition=action[i])

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
        return self._get_observation()

    def _get_observation(self):
        # Example: get joint states
        joint_states = p.getJointStates(self.humanoidId, range(p.getNumJoints(self.humanoidId)))
        joint_positions = [state[0] for state in joint_states]
        return np.array(joint_positions, dtype=np.float32)

    def _calculate_reward(self, observation):
        # Reward for staying upright
        base_position, _ = p.getBasePositionAndOrientation(self.humanoidId)
        upright_reward = base_position[2]  # Higher reward for staying upright

        # Penalize large joint angles
        joint_penalty = -np.linalg.norm(observation)

        return upright_reward + joint_penalty

    def _check_done(self, observation):
        # Example: check if robot has fallen
        base_position, _ = p.getBasePositionAndOrientation(self.humanoidId)
        return base_position[2] < 0.5  # Consider done if the robot falls

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect() 