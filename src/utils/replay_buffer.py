# common
import os
import numpy as np
import h5py
from collections import deque

from dm_env import StepType
from src import dmc


step_type_lookup = {0: StepType.FIRST, 1: StepType.MID, 2: StepType.LAST}


class ReplayBuffer(object):
    def __init__(self, buffer_size, nstep, discount, frame_stack):
        self.buffer_size = buffer_size
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))
        self.next_dis = discount**nstep

        # self.offline_dir is used to determine tasks
        # self.degrees is used to determine rotation angles
        # self.offline_dir = offline_dir
        self._degrees = np.zeros([self.buffer_size], dtype=np.float32)

    @property
    def degrees(self):
        return self._degrees

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.act_shape = time_step.action.shape

        self.obs = np.zeros(
            [self.buffer_size, self.ims_channels, *self.obs_shape[1:]],
            dtype=np.uint8,
        )
        self.act = np.zeros(
            [self.buffer_size, *self.act_shape], dtype=np.float32
        )
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)
        self.step_type = np.zeros([self.buffer_size], dtype=np.float32)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def sample(self, batch_size, get_degrees=False):
        indices = np.random.choice(self.valid.nonzero()[0], size=batch_size)
        return self._gather_nstep_indices(indices, get_degrees=get_degrees)

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)

        self._add_data_point(time_step)

    def _add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels :]

        if first:
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index : self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index : end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index : self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index : end_index] = latest_obs
                self.valid[self.index : end_invalid] = False
            self.index = end_index
            self.traj_index = 1

        else:
            # Check most recent image
            np.copyto(self.obs[self.index], latest_obs)
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.step_type[self.index] = time_step.step_type

            self.valid[(self.index + self.frame_stack) % self.buffer_size] = (
                False
            )
            if self.traj_index >= self.nstep:
                self.valid[
                    (self.index - self.nstep + 1) % self.buffer_size
                ] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def _gather_nstep_indices(self, indices, sarsa=False, get_degrees=False):
        n_samples = indices.shape[0]
        all_gather_ranges = (
            np.stack(
                [
                    np.arange(
                        indices[i] - self.frame_stack, indices[i] + self.nstep
                    )
                    for i in range(n_samples)
                ],
                axis=0,
            )
            % self.buffer_size
        )

        gather_ranges = all_gather_ranges[:, self.frame_stack :]
        obs_gather_ranges = all_gather_ranges[:, : self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack :]

        all_rewards = self.rew[gather_ranges]

        # Could implement below operation as a matmul in pytorch for
        # marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(
            self.obs[obs_gather_ranges], [n_samples, *self.obs_shape]
        )
        nobs = np.reshape(
            self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape]
        )
        act = self.act[indices]
        dis = np.expand_dims(
            self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1
        )

        if sarsa:
            nact = self.act[indices + self.nstep]
            return (obs, act, rew, dis, nobs, nact)
        elif get_degrees:
            degrees = self.degrees[indices]
            return (obs, act, rew, dis, nobs, degrees)
        else:
            return (obs, act, rew, dis, nobs)


def load_offline_dataset_into_buffer(
    offline_dirs, replay_buffer, frame_stack, replay_buffer_size, verbose=True
):
    if not isinstance(offline_dirs, list):
        raise TypeError(f"dirrectory must be a list, got {type(offline_dirs)}")

    for offline_dir in offline_dirs:
        assert os.path.exists(offline_dir), f"{offline_dir} does not exist"

        filenames = sorted(offline_dir.glob("*.hdf5"))
        num_steps = 0
        for filename in filenames:
            try:
                episodes = h5py.File(filename, "r")
                episodes = {k: episodes[k][:] for k in episodes.keys()}
                _add_offline_data_to_buffer(
                    episodes, replay_buffer, framestack=frame_stack
                )
                length = episodes["reward"].shape[0]
                num_steps += length
            except Exception as e:
                print(f"Could not load episode {str(filename)}: {e}")
                continue

            if verbose:
                print(
                    f"Loaded {num_steps} offline timesteps so far..."
                )
            if num_steps >= replay_buffer_size:
                break

        if verbose:
            print("Finished, loaded {} timesteps.".format(int(num_steps)))


def _add_offline_data_to_buffer(offline_data, replay_buffer, framestack=3):
    offline_data_length = offline_data["reward"].shape[0]

    for v in offline_data.values():
        assert v.shape[0] == offline_data_length

    for idx in range(offline_data_length):
        time_step = _get_timestep_from_idx(offline_data, idx)
        if time_step.first():
            stacked_frames = deque(maxlen=framestack)
            while len(stacked_frames) < framestack:
                stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(
                observation=np.concatenate(stacked_frames, axis=0)
            )
            replay_buffer.add(time_step_stack)
        else:
            stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(
                observation=np.concatenate(stacked_frames, axis=0)
            )
            replay_buffer.add(time_step_stack)


def _get_timestep_from_idx(offline_data: dict, idx: int):
    return dmc.ExtendedTimeStep(
        step_type=step_type_lookup[offline_data["step_type"][idx]],
        reward=offline_data["reward"][idx],
        observation=offline_data["observation"][idx],
        discount=offline_data["discount"][idx],
        action=offline_data["action"][idx],
    )
