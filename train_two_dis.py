# common
from pathlib import Path
import hydra
import numpy as np
import os
import wandb
from omegaconf import OmegaConf
from datetime import datetime
from joblib import Parallel, delayed


# deep RL
import torch


# own
from src import dmc
from src.utils import (
    Timer,
    Until,
    Every,
    auto_select_gpu,
    set_seed_everywhere,
    VideoRecorder,
    Logger,
    ReplayBuffer,
    load_offline_dataset_into_buffer,
)
from src.train_common import make_agent, log_bias_evaluation
from src.eval_common import log_bellman, run_eval


torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.task_name = (
            f"{self.cfg.task.domain}_{self.cfg.task.task}_{self.cfg.task.type}"
        )
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.eval_on_distracting = self.cfg.eval_on_distracting

        # create logger
        self.logger = Logger(self.work_dir)

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            task_name=self.task_name,
        )

        # setup env and agent
        self.setup_agent()

        self.timer = Timer()
        self._global_step = 1

        # run with wandb logging
        OmegaConf.save(cfg, f"{self.work_dir}/config.yaml")
        if not cfg.no_track:
            wandb.init(
                project="pytorch.vd4rl_new",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=str(os.path.basename(self.work_dir)),
            )

    def setup_agent(self):
        # create envs
        eval_env = dmc.make(
            self.cfg.task.domain,
            self.cfg.task.task,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            distracting_mode=None,
        )

        # create agent
        self.agent = make_agent(
            eval_env.observation_spec(),
            eval_env.action_spec(),
            self.cfg.device,
            self.cfg,
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self, save_video=False, alpha=0.2, gamma=0.99, n_mc_cutoff=150):
        if self.eval_on_distracting:
            distraction_modes = [
                "easy",
                "medium",
                "hard",
                "fixed_easy",
                "fixed_medium",
                "fixed_hard",
                None,
            ]
        else:
            distraction_modes = [None]

        for mode in distraction_modes:
            results = Parallel(n_jobs=self.cfg.eval_parallel_jobs)(
                delayed(run_eval)(
                    self.cfg.task.domain,
                    self.cfg.task.task,
                    self.cfg.frame_stack,
                    self.cfg.action_repeat,
                    self.cfg.seed,
                    mode,
                    self.agent,
                    gamma,
                    alpha,
                    self.video_recorder,
                    (episode == 0 and save_video),
                    self.global_step,
                    n_mc_cutoff,
                )
                for episode in range(self.cfg.eval_episodes)
            )

            # clear remaining jobs
            from joblib.externals.loky import get_reusable_executor

            get_reusable_executor(timeout=10).shutdown(wait=True)

            # for calculating bias
            (
                total_reward_list,
                final_mc_entropy_list,
                final_obs_list,
                final_act_list,
                steps,
            ) = zip(*results)

            # logging
            if mode is None:
                # for normal env
                with self.logger.log_and_dump_ctx(
                    self.global_frame, ty="eval"
                ) as log:
                    log("episode_reward_mean", np.mean(total_reward_list))
                    log("episode_reward_std", np.std(total_reward_list))
                    log(
                        "episode_length",
                        np.mean(steps) * self.cfg.action_repeat,
                    )
                    log("step", self.global_step)
                if not self.cfg.no_track:
                    wandb.log(
                        {
                            "step": self.global_step,
                            "normal_test_reward_mean": np.mean(
                                total_reward_list
                            ),
                            "normal_test_reward_std": np.std(
                                total_reward_list
                            ),
                        }
                    )
            else:
                # for distracting envs
                self.logger.log(
                    f"eval/{mode}_episode_reward_mean",
                    np.mean(total_reward_list),
                    self.global_frame,
                )
                if not self.cfg.no_track:
                    wandb.log(
                        {
                            "step": self.global_step,
                            f"{mode}_test_reward_mean": np.mean(
                                total_reward_list
                            ),
                            f"{mode}_test_reward_std": np.std(
                                total_reward_list
                            ),
                        }
                    )

            if self.cfg.calculate_bias and not self.cfg.no_track:
                # calculate bias
                log_bias_evaluation(
                    final_obs_list,
                    final_act_list,
                    final_mc_entropy_list,
                    self.agent,
                    self.device,
                    mode,
                    self.global_step,
                )

    def train_offline(self):
        # create replay buffer
        self.replay_buffer = ReplayBuffer(
            self.cfg.replay_buffer_size,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.frame_stack,
        )
        self.replay_buffer_dis = ReplayBuffer(
            self.cfg.replay_buffer_size,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.frame_stack,
        )

        # Open dataset, load as memory buffer: original
        print(f"loading dataset from {self.cfg.offline_dir}...")
        load_offline_dataset_into_buffer(
            [Path(self.cfg.offline_dir)],
            self.replay_buffer,
            self.cfg.frame_stack,
            self.cfg.replay_buffer_size,
        )

        # Open dataset, load as memory buffer: distracting
        print()
        print(
            f"loading distracting dataset: {self.cfg.task.mode} from "
            f"{self.cfg.offline_dir_dis}..."
        )
        load_offline_dataset_into_buffer(
            [Path(self.cfg.offline_dir_dis)],
            self.replay_buffer_dis,
            self.cfg.frame_stack,
            self.cfg.replay_buffer_size,
        )

        print("-" * 40)
        print()

        if self.replay_buffer.index == -1:
            raise ValueError("No offline data loaded, check directory.")

        # predicates
        train_until_step = Until(self.cfg.num_timesteps + 1, 1)
        eval_freq = Every(self.cfg.eval_freq, 1)
        show_stats_freq = Every(self.cfg.show_stats_freq, 1)

        metrics = {}
        elapsed_time, total_time = self.timer.reset()
        local_step = 0
        while train_until_step(self.global_step):
            # try to update the agent
            metrics_curr = self.agent.update(
                self.replay_buffer,
                self.replay_buffer_dis,
                self.global_step,
                self.cfg.batch_size,
            )
            metrics.update(metrics_curr)

            if show_stats_freq(self.global_step):
                # wait until all the metrics schema is populated

                for key, value in metrics.items():
                    self.logger.log(f"train/{key}", value, self.global_frame)

                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", local_step / elapsed_time)
                        log("total_time", total_time)
                        log(
                            "ETA",
                            (
                                total_time
                                / self.global_step
                                * (self.cfg.num_timesteps - self.global_step)
                            ),
                        )
                        log(
                            "endtime",
                            (
                                total_time
                                / self.global_step
                                * (self.cfg.num_timesteps - self.global_step)
                            ),
                        )
                        log("buffer_size", len(self.replay_buffer))
                        log("step", self.global_step)

                    metrics.update(
                        {
                            "fps": local_step / elapsed_time,
                            "total_time": total_time,
                            "step": self.global_step,
                        }
                    )

                    if not self.cfg.no_track:
                        wandb.log(metrics)
                    metrics = {}
                    local_step = 0

            # try to evaluate
            if eval_freq(self.global_step):
                # log norms
                norms = self.agent.calculate_norms()
                norms.update({"step": self.global_step})
                if not self.cfg.no_track:
                    wandb.log(norms)

                # log bellman distribution historgram
                if self.cfg.plot_bellman:
                    log_bellman(
                        self.agent,
                        self.replay_buffer,
                        self.replay_buffer_dis,
                        self.global_step,
                        self.work_dir,
                        self.cfg,
                    )

                # eval
                self.agent.train(training=False)
                self.logger.log(
                    "eval_total_time",
                    self.timer.total_time(),
                    self.global_frame,
                )

                self.eval(save_video=True)
                self.agent.train(training=True)
                self.agent.save_models(self.global_step, self.work_dir)
                elapsed_time, total_time = self.timer.reset()

            local_step += 1
            self._global_step += 1


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg):
    print("=" * 65)
    print()
    start_time = datetime.now()
    start_time = start_time.strftime("%Y/%m/%d %H:%M:%S")
    print(f"start time: {start_time}")

    # set gpus
    autogpu = auto_select_gpu(force_select=cfg.gpu)
    cfg.device = f"cuda:{autogpu}"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = f"{autogpu}"
    os.environ["EGL_DEVICE_ID"] = f"{autogpu}"

    # directory checks
    # if one of these directories are specified, expect both to be specified
    if cfg.offline_dir != "..." and cfg.offline_dir_dis == "...":
        raise Exception(
            "Cannot specific one path but not the other. "
            "Please specify both cfg.offline_dir and cfg.offline_dir_dis."
        )
    if cfg.offline_dir == "..." and cfg.offline_dir_dis != "...":
        raise Exception(
            "Cannot specific one path but not the other. "
            "Please specify both cfg.offline_dir and cfg.offline_dir_dis."
        )

    # force using medium if not specified
    # ignores task.mode
    if cfg.offline_dir == "...":        # only overwrite if not specified
        cfg.offline_dir = os.path.join(
            f"{os.path.expanduser('~')}/.v_d4rl/vd4rl/distracting/",
            f"{cfg.task.domain}_{cfg.task.task}",
            cfg.task.type,
            "84px",
            "medium",
        )
    if cfg.offline_dir_dis == "...":    # only overwrite if not specified
        cfg.offline_dir_dis = os.path.join(
            f"{os.path.expanduser('~')}/.v_d4rl/vd4rl/distracting/",
            f"{cfg.task.domain}_{cfg.task.task}",
            cfg.task.type,
            "84px",
            cfg.task.mode,
        )

    # if uses extended collected dataset: replace the directories
    # first, remove the "extended" in "task.mode"
    if cfg.extended_dataset:
        cfg.offline_dir = cfg.offline_dir.replace("_extended", "")
        cfg.offline_dir = cfg.offline_dir.replace("vd4rl", "vd4rl_extended")
        cfg.offline_dir_dis = cfg.offline_dir_dis.replace("_extended", "")
        cfg.offline_dir_dis = cfg.offline_dir_dis.replace(
            "vd4rl", "vd4rl_extended"
        )

    # setup
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    print("-" * 40)
    print()

    # run
    workspace.train_offline()
    if not cfg.no_track:
        wandb.finish()

    end_time = datetime.now()
    end_time = end_time.strftime("%Y/%m/%d %H:%M:%S")
    print(f"workspace: {root_dir}")
    print(f"start time: {start_time}")
    print(f"end time: {end_time}")


if __name__ == "__main__":
    main()
