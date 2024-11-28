# Environment wrapper used for Temporally Correlated Episodic (TCE) RL

from typing import Tuple, Optional, Callable, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.core import ObsType

from fancy_gym.black_box.controller.base_controller import BaseController
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.utils import get_numpy

import copy


class TCEWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env: RawInterfaceWrapper,
                 tracking_controller: BaseController,
                 verbose: int = 1,
                 learn_sub_trajectories: bool = False,
                 replanning_schedule: Optional[callable] = None,
                 reward_aggregation: callable = np.sum):
        """
        gym.Wrapper for traj-in traj-out wrapper

        Args:
            env: The (wrapped) environment this wrapper is applied on
            tracking_controller: Translates the desired trajectory to raw action sequences
            verbose: level of detail for returned values in info dict.
            learn_sub_trajectories: Transforms full episode learning into learning sub-trajectories, similar to
                step-based learning
            replanning_schedule: callable that receives
            reward_aggregation: function that takes the np.ndarray of step rewards as input and returns the trajectory
                reward, default summation over all values.
        """
        super().__init__(env)

        # Time steps
        self.current_traj_steps = 0

        # In case of replanning
        self.do_replanning = replanning_schedule is not None
        self.replanning_schedule = replanning_schedule or (lambda *x: False)

        # trajectory follower
        self.tracking_controller = tracking_controller

        # state and action spaces
        self.num_dof = self._get_num_dof()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        # initial condition
        self.init_time = 0
        self.init_pos = self.env.get_wrapper_attr('current_pos')
        self.init_vel = self.env.get_wrapper_attr('current_vel')

        # reward computation
        self.reward_aggregation = reward_aggregation

        # rendering
        self.do_render = False
        self.verbose = verbose

    def _get_num_dof(self):
        return self.env.action_space.shape[0]

    def _get_observation_space(self):
        assert isinstance(self.env.observation_space, Box)

        # Append init time space
        init_time_low = 0.0
        dt = self.env.get_wrapper_attr('dt')
        init_time_high = self.env.spec.max_episode_steps * dt
        low = np.append(self.env.observation_space.low, init_time_low)
        high = np.append(self.env.observation_space.high, init_time_high)

        # Append init pos space
        init_pos_low = np.full([self.num_dof], -np.inf)
        init_pos_high = np.full([self.num_dof], np.inf)
        low = np.append(low, init_pos_low)
        high = np.append(high, init_pos_high)

        # Append init vel space
        init_vel_low = np.full([self.num_dof], -np.inf)
        init_vel_high = np.full([self.num_dof], np.inf)
        low = np.append(low, init_vel_low)
        high = np.append(high, init_vel_high)

        observation_space = Box(low, high,
                                dtype=self.env.observation_space.dtype)
        return observation_space

    def _get_action_space(self):
        action_bounds = np.zeros([2, self.num_dof * 2])
        action_bounds[0] = -np.inf
        action_bounds[1] = np.inf
        action_space = Box(low=action_bounds[0],
                           high=action_bounds[1],
                           dtype=self.env.action_space.dtype)
        return action_space

    def observation(self, observation):
        observation = np.concatenate([observation, np.asarray([self.init_time]),
                                      self.init_pos, self.init_vel], axis=-1)
        return observation.astype(self.observation_space.dtype)

    def get_initial_condition(self) -> Tuple:
        return copy.deepcopy((self.init_time, self.init_pos, self.init_vel))

    def get_trajectory(self, action: np.ndarray) -> Tuple:
        """
        Split position and velocity
        Args:
            action: coupled position and velocity

            Shape of action:
            [num_times, num_dof]

        Returns:
            pos_traj: position trajectory
            vel_traj: velocity trajectory
        """
        pos_traj = action[..., :self.num_dof]
        vel_traj = action[..., self.num_dof:]

        return pos_traj, vel_traj

    def step(self, action: np.ndarray):

        # Get initial condition
        segment_init_time, segment_init_pos, segment_init_vel = \
            self.get_initial_condition()

        # Get trajectory to follow
        step_desired_pos, step_desired_vel = self.get_trajectory(action)
        num_times = len(step_desired_pos)

        # Actual pos and vel
        step_actual_pos = np.zeros_like(step_desired_pos)
        step_actual_vel = np.zeros_like(step_desired_vel)

        # Initialize segment storage
        step_states = np.zeros([num_times, *self.observation_space.shape],
                               dtype=self.env.observation_space.dtype)
        step_actions = np.zeros([num_times, *self.env.action_space.shape])
        step_rewards = np.zeros(num_times)
        step_terminations = np.zeros(num_times, dtype=bool)
        step_truncations = np.zeros(num_times, dtype=bool)

        terminated, truncated = False, False
        segment_info = dict()
        t = 0

        traj_valid = True

        # Low level control loop
        for t, (pos, vel) in enumerate(zip(step_desired_pos, step_desired_vel)):
            # Check if trajectory is valid
            if hasattr(self.env, "check_traj_step_validity"):
                is_valid = self.env.check_traj_step_validity(pos)
            else:
                is_valid = True

            if not is_valid:
                traj_valid = False

            # Update time steps
            self.current_traj_steps += 1

            # Update initial conditions
            dt = self.env.get_wrapper_attr('dt')
            self.init_time = self.current_traj_steps * dt
            self.init_pos = pos
            self.init_vel = vel

            # Get low-level action
            step_action = self.tracking_controller.get_action(
                pos, vel, self.env.get_wrapper_attr('current_pos'),
                self.env.get_wrapper_attr('current_vel'))
            step_actual_pos[t] = self.env.get_wrapper_attr('current_pos')
            step_actual_vel[t] = self.env.get_wrapper_attr('current_vel')

            clipped_step_action = np.clip(step_action,
                                          self.env.action_space.low,
                                          self.env.action_space.high)

            # Step
            state, reward, terminated, truncated, info = (
                self.env.step(clipped_step_action))
            # Apply penalty if trajectory violates constraints
            if not traj_valid:
                reward = self.env.get_invalid_traj_step_penalty(pos)

            # Storage state, action, reward and done
            state = self.observation(state)
            step_states[t] = state
            step_actions[t] = clipped_step_action
            step_rewards[t] = reward
            step_terminations[t] = terminated
            step_truncations[t] = truncated

            # Storage info
            for k, v in info.items():
                elems = segment_info.get(k, [None] * num_times)
                elems[t] = v
                segment_info[k] = elems

            # Render
            if self.do_render:
                self.env.render()

            # Stop or replanning
            if terminated or truncated:
                break

        # Truncate storage if early stopping or replanning
        step_states = step_states[:t + 1]
        step_actual_pos = step_actual_pos[:t + 1]
        step_actual_vel = step_actual_vel[:t + 1]

        step_actions = step_actions[:t + 1]
        step_rewards = step_rewards[:t + 1]
        step_terminations = step_terminations[:t + 1]
        step_truncations = step_truncations[:t + 1]

        segment_info.update({k: v[:t + 1] for k, v in segment_info.items()})

        # The new initial conditions
        segment_end_time, segment_end_desired_pos, segment_end_desired_vel \
            = self.get_initial_condition()

        # Save stepwise info, shape [num_times, *dim_data]
        segment_info["step_desired_pos"] = step_desired_pos
        segment_info["step_desired_vel"] = step_desired_vel

        segment_info["step_actual_pos"] = step_actual_pos
        segment_info["step_actual_vel"] = step_actual_vel

        segment_info["step_states"] = step_states
        segment_info["step_actions"] = step_actions
        segment_info["step_rewards"] = step_rewards
        segment_info["step_terminations"] = step_terminations
        segment_info["step_truncations"] = step_truncations

        # Save segmentwise info, shape [*dim_data]
        segment_info['segment_init_time'] = segment_init_time
        segment_info['segment_init_pos'] = segment_init_pos
        segment_info['segment_init_vel'] = segment_init_vel
        segment_info['segment_end_time'] = segment_end_time
        segment_info['segment_end_desired_pos'] = segment_end_desired_pos
        segment_info['segment_end_desired_vel'] = segment_end_desired_vel
        segment_info['segment_reward'] = self.reward_aggregation(step_rewards)
        segment_info["segment_length"] = t + 1

        # Return state, reward, done of the last step
        # Return info dict containing both step- and segmentwise info
        return (state, segment_info['segment_reward'], terminated, truncated,
                segment_info)

    def render(self):
        self.do_render = True

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        self.current_traj_steps = 0
        obs, info = super().reset(seed=seed, options=options)

        self.init_time = 0
        self.init_pos = self.env.get_wrapper_attr('current_pos')
        self.init_vel = self.env.get_wrapper_attr('current_vel')

        # Use the latest initial state
        return self.observation(obs[:-self.num_dof * 2 - 1]), info
