
import sys
sys.path.insert(0,'uav_trajectories/scripts')
from uav_trajectory import Trajectory


import numpy as np


trajectory = Trajectory()
trajectory.loadcsv("uav_trajectories/traj_change_y.csv")

e = trajectory.eval(0.5)
print("Position:", e.pos)
print("Velocity:", e.vel)
print("Acceleration:", e.acc)
print("Omega:", e.omega)
print("Yaw:", e.yaw)
print("Roll:", e.roll)
print("Pitch:", e.pitch)