import time
import warnings
from typing import TypeVar

import einops
import numpy as np
import torch
import rospy
import actionlib
import tf.transformations
from diff_planner.config import load_diff_model
from diff_planner.diffuser.utils import apply_dict
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from diff_planner_msgs.msg import PlanPathAction


class BaseDiffusionPlanner(object):
    def __init__(self, horizon, dt_plan, state_dim, action_dim, device, t_start=None):
        self.device = device
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt_plan = dt_plan
        self.plan = torch.empty((horizon, state_dim), device=device)
        self.action = torch.empty((horizon, action_dim), device=device)
        self.plan_t_start = t_start if t_start is not None else time.time_ns()
        self.obs_indices = set()

    def add_observation(self, obs, t=None):
        obs_index = self.get_index(t)
        if obs_index >= self.horizon:
            return
        self.plan[obs_index] = obs
        self.obs_indices.add(obs_index)

    def generate_plan(self, start, end, start_time=None):
        self.plan = torch.empty((self.horizon, self.state_dim), device=self.device)
        self.action = torch.empty((self.horizon, self.action_dim), device=self.device)
        self.plan[-1] = end
        self.obs_indices = {-1}

        self.set_plan_start(start_time)
        self.add_observation(start, self.plan_t_start)

        self.update_plan()

    def update_plan(self):
        raise NotImplementedError

    def get_setpoint(self, t=None):
        action_index = self.get_index(t)
        print(f'Getting action {action_index}')
        if action_index >= self.horizon:
            warnings.warn('No planned action')
            return None, None
        return self.action[action_index], action_index == self.horizon-1

    def get_index(self, t=None):
        if t is None:
            t = time.time_ns()
        return int((t - self.plan_t_start) / (self.dt_plan * 1e9))

    def set_plan_start(self, t=None):
        self.plan_t_start = t if t is not None else time.time_ns()


class MockPlanner(BaseDiffusionPlanner):
    def update_plan(self):
        self.plan[...] = self.plan[-1]
        self.action[...] = self.plan[-1]


class GausInvDynPlanner(BaseDiffusionPlanner):
    def __init__(self, model, horizon, dt_plan, dt_sample, state_dim, action_dim, device, mode='pos'):
        super().__init__(horizon, dt_plan, state_dim, action_dim, device)
        if mode not in ['pos', 'vel']:
            raise AttributeError('Invalid action mode.')
        self.mode = mode
        self.model = model

        assert dt_sample >= dt_plan
        self.dt_sample = dt_sample
        self.sample_step_index = int(np.round(dt_sample / dt_plan))

    def update_plan(self):
        self.sample_plan()
        if self.mode == 'pos':
            self.action[:-self.sample_step_index] = self.plan[1:self.horizon-self.sample_step_index+1, :7]
            self.action[-self.sample_step_index:] = self.plan[-1, :7]
        else:
            raise NotImplementedError
            # self.action[now:-self.sample_step_index] = self.plan[now + 1:, 7:13]
            # self.action[-self.sample_step_index:] = self.plan[-1, :7]

    def sample_plan(self):
        print('Starting sample:')
        print('Normalizing...')
        norm_plan = self.model.normalizer.normalize(self.plan, 'observations')
        print('Getting conditions...')
        conditions = {i: norm_plan[i][None, :] for i in self.obs_indices}

        conditions = apply_dict(
            einops.repeat,
            conditions,
            'b d -> (repeat b) d', repeat=1,
        )
        print('Forward pass...')
        norm_plan = self.model.conditional_sample(conditions, horizon=self.horizon)
        print('Unnormalizing...')
        self.plan = self.model.normalizer.unnormalize(norm_plan, 'observations')[0, ...]


DiffPlanner = TypeVar('DiffPlanner', bound=BaseDiffusionPlanner)


def pose_ros2pytorch(pose_stamped: PoseStamped, **kwargs):
    pose = torch.empty((7,), **kwargs)

    pose[0] = pose_stamped.pose.position.x
    pose[1] = pose_stamped.pose.position.y
    pose[2] = pose_stamped.pose.position.z

    pose[3] = pose_stamped.pose.orientation.x
    pose[4] = pose_stamped.pose.orientation.y
    pose[5] = pose_stamped.pose.orientation.z
    pose[6] = pose_stamped.pose.orientation.w
    return pose


def pose_pytorch2ros(pose: torch.Tensor, counter: int):
    pose_stamped = PoseStamped()
    pose_stamped.header.seq = counter
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.header.frame_id = "global"

    pose_stamped.pose.position.x = pose[0]
    pose_stamped.pose.position.y = pose[1]
    pose_stamped.pose.position.z = pose[2]

    pose_stamped.pose.orientation.x = pose[3]
    pose_stamped.pose.orientation.y = pose[4]
    pose_stamped.pose.orientation.z = pose[5]
    pose_stamped.pose.orientation.w = pose[6]
    return pose_stamped


def pose_franka2pytorch(msg, **kwargs):
    quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
    quaternion = quaternion / np.linalg.norm(quaternion)

    pose = torch.empty((7,), **kwargs)
    pose[0] = msg.O_T_EE[12]
    pose[1] = msg.O_T_EE[13]
    pose[2] = msg.O_T_EE[14]
    pose[3] = quaternion[0]
    pose[4] = quaternion[1]
    pose[5] = quaternion[2]
    pose[6] = quaternion[3]
    return pose


class PosePlanner(object):
    def __init__(self, planner: DiffPlanner):
        rospy.init_node('diff_planner', anonymous=True)
        self.planner = planner
        self.dt_plan = planner.dt_plan
        self.dt_sample = self.dt_plan if not hasattr(planner, 'dt_sample') else planner.dt_sample

        self.sub = rospy.Subscriber('franka_state_controller/franka_states', FrankaState, self.observation_cb)
        self.pub = rospy.Publisher('setpoint', PoseStamped, queue_size=1)

        self.last_pose = torch.empty((7,), device=self.planner.device)
        self.setpoint = torch.empty((7,), device=self.planner.device)
        self.goal = torch.empty((7,), device=self.planner.device)
        self.counter = 0
        self.final = False

        self._as = actionlib.SimpleActionServer('plan', PlanPathAction,
                                                execute_cb=self.plan_cb, auto_start=False)
        self._as.start()

        self.wait_for_initial_pose()
        rospy.Timer(rospy.Duration(self.dt_plan), self.publish_cb)
        rospy.spin()

    def observation_cb(self, obs):
        pose = pose_franka2pytorch(obs, device=self.planner.device)
        self.planner.add_observation(pose)
        self.last_pose = pose

    def publish_cb(self, *args):
        setpoint = pose_pytorch2ros(self.setpoint, self.counter)
        #rospy.loginfo(f'Publishing: {setpoint}')
        self.pub.publish(setpoint)
        self.counter += 1

    def plan_cb(self, goal):
        r = rospy.Rate(int(np.round(1/self.dt_sample)))
        self.goal = pose_ros2pytorch(goal.goal_pose)

        rospy.loginfo(f'Generating plan...')
        tstart = rospy.Time().now()
        self.planner.generate_plan(self.last_pose, self.goal)
        duration = rospy.Time().now() - tstart
        rospy.loginfo(f'Finished plan generation in {(duration.secs + duration.nsecs/1e9):.3f} seconds')

        self.final = False
        self.planner.set_plan_start()
        while not self.final and not rospy.is_shutdown():
            setpoint = self.next_pose()
            if setpoint is None:
                continue
            self.update_setpoint(setpoint)
            r.sleep()
        self._as.set_aborted()

    def update_setpoint(self, new):
        self.setpoint = new

    def next_pose(self):
        if self.final:
            rospy.loginfo('Planner reached final state')
            return None

        setpoint, self.final = self.planner.get_setpoint()

        if setpoint is None:
            rospy.logwarn('No setpoint available')
            self.final = True
            return None
        return setpoint

    def wait_for_initial_pose(self):
        msg = rospy.wait_for_message("franka_state_controller/franka_states", FrankaState)
        self.setpoint = pose_franka2pytorch(msg)


def run_node():
    config_file = rospy.get_param('config_file', '')
    pt_file = rospy.get_param('pt_file', '')
    mock = rospy.get_param('mock', True)

    if mock:
        planner = PosePlanner(MockPlanner(10, 0.08, 7, 7, 'cuda'))
    else:
        model = load_diff_model(config_file, pt_file)
        planner = PosePlanner(
            GausInvDynPlanner(model, model.horizon, 0.08, 0.08, 7, 7, 'cuda')
        )


def main():
    try:
        run_node()
    except rospy.ROSInterruptException:
        pass

