"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask
from collections import deque

class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target = np.array([0.0,0.0,10.0])  # target coord
        self.dist_error = 1.0
        #for velocity
        self.position_que = deque(maxlen=2)
        self.time_que = deque(maxlen=2)
        self.zero_vel = np.zeros(3)
    
    def reset(self):
        self.position_que = deque(maxlen=2)
        rand_z = np.random.normal(2.0, 0.1)
        self.position_que.append(np.array([0.0, 0.0, rand_z]))
        self.time_que = deque(maxlen=2)
        self.time_que.append(0.0)
        
        return Pose(
                position=Point(0.0, 0.0, rand_z),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector
        #state = np.array([
        #        pose.position.x, pose.position.y, pose.position.z,
        #        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        #lin_acc = np.array([linear_acceleration.x,linear_acceleration.y,linear_acceleration.z])
        self.position_que.append(pos)
        self.time_que.append(timestamp)
        delta_t = self.time_que[1]-self.time_que[0]
        delta_s = self.position_que[1] - self.position_que[0]
        if delta_t == 0.0:
            vel = self.zero_vel
        else:
            vel = delta_s/delta_t
        
        state = np.concatenate((pos, vel, self.target), axis=-1)
        

        # Compute reward / penalty and check if this episode is complete
        done = False
        distance = np.linalg.norm(state[:3]-self.target)
        #reward = -min(distance,20.)
        reward = -10.0 + ((30./(1. + distance)))
        #if pos[2] > 1.0 and pos[2] < 15.0:
        #    reward -= 0.01* np.linalg.norm(vel)
        #else:
        #    reward -=5.0
        #reward = (8. - distance)*0.5
        
        if timestamp > self.max_duration:
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
