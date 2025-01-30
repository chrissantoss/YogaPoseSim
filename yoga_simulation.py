import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
physicsClient = p.connect(p.GUI)  # Use p.DIRECT for non-GUI mode

# Set additional search path to find URDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and humanoid robot
planeId = p.loadURDF("plane.urdf")
humanoidId = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 1], useFixedBase=False)

# Set gravity
p.setGravity(0, 0, -9.81)

# Simulation parameters
timeStep = 1.0 / 240.0
p.setTimeStep(timeStep)

# Run the simulation
try:
    while True:
        p.stepSimulation()
        time.sleep(timeStep)
except KeyboardInterrupt:
    pass

# Disconnect from PyBullet
p.disconnect() 