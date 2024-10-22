import os
import argparse
import datetime
import contextlib
import joblib
import numpy as np
import torch
from functools import partial
from typing import Optional
from tqdm import tqdm as std_tqdm


# own
from src.models import Encoder, ActorSimple, ActorDefault
from src.utils import schedule

# auto width change tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)


# define distracting modes to evaluate on
distracting_modes = [
    None,
    "easy",
    "medium",
    "hard",
    "easy_background_dis_only",
    "easy_camera_dis_only",
    "easy_colour_dis_only",
    "medium_background_dis_only",
    "medium_camera_dis_only",
    "medium_colour_dis_only",
    "hard_background_dis_only",
    "hard_camera_dis_only",
    "hard_colour_dis_only",
]
kwargs_dict = {
    "easy_background_dis_only": {
        "difficulty": "easy",
        "fixed_distraction": False,
        "apply_background_dis": True,
        "apply_camera_dis": False,
        "apply_color_dis": False,
    },
    "easy_camera_dis_only": {
        "difficulty": "easy",
        "fixed_distraction": False,
        "apply_background_dis": False,
        "apply_camera_dis": True,
        "apply_color_dis": False,
    },
    "easy_colour_dis_only": {
        "difficulty": "easy",
        "fixed_distraction": False,
        "apply_background_dis": False,
        "apply_camera_dis": False,
        "apply_color_dis": True,
    },
    "medium_background_dis_only": {
        "difficulty": "medium",
        "fixed_distraction": False,
        "apply_background_dis": True,
        "apply_camera_dis": False,
        "apply_color_dis": False,
    },
    "medium_camera_dis_only": {
        "difficulty": "medium",
        "fixed_distraction": False,
        "apply_background_dis": False,
        "apply_camera_dis": True,
        "apply_color_dis": False,
    },
    "medium_colour_dis_only": {
        "difficulty": "medium",
        "fixed_distraction": False,
        "apply_background_dis": False,
        "apply_camera_dis": False,
        "apply_color_dis": True,
    },
    "hard_background_dis_only": {
        "difficulty": "hard",
        "fixed_distraction": False,
        "apply_background_dis": True,
        "apply_camera_dis": False,
        "apply_color_dis": False,
    },
    "hard_camera_dis_only": {
        "difficulty": "hard",
        "fixed_distraction": False,
        "apply_background_dis": False,
        "apply_camera_dis": True,
        "apply_color_dis": False,
    },
    "hard_colour_dis_only": {
        "difficulty": "hard",
        "fixed_distraction": False,
        "apply_background_dis": False,
        "apply_camera_dis": False,
        "apply_color_dis": True,
    },
}


@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs):
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()


def act(encoder, actor, obs, step, stddev_schedule):
    obs = torch.as_tensor(obs).cuda().float()
    _, obs = encoder(obs.unsqueeze(0))
    stddev = schedule(stddev_schedule, step)
    dist = actor(obs, stddev)
    action = dist.mean
    return action.cpu().numpy()[0]


def _run_eval(
    domain,
    task,
    distracting_mode,
    encoder,
    actor,
    stddev_schedule,
    kwargs_dict=None,
):
    from src import dmc

    eval_env = dmc.make(
        domain,
        task,
        3,
        2,
        0,
        distracting_mode,
        kwargs_dict=kwargs_dict if distracting_mode in kwargs_dict else None,
    )
    time_step = eval_env.reset()

    total_reward = 0.0
    step = 0.0

    while not time_step.last():
        eval_obs = time_step.observation
        eval_obs = eval_obs / 255.0

        with torch.no_grad():
            action = act(
                encoder,
                actor,
                eval_obs,
                step=1000000.0,
                stddev_schedule=stddev_schedule,
            )
        time_step = eval_env.step(action)
        total_reward += time_step.reward
        step += 1

    return total_reward


def load_weights_and_eval_one_mode(
    saved_weights_dir, n_jobs, distracting_mode, algo,
):

    # common settings
    test_episodes = 30
    feature_dim = 50
    hidden_dim = 1024
    stddev_schedule = "linear(1.0,0.1,250000)"
    domain = "cheetah"
    task = "run"

    from src import dmc

    dummy_env = dmc.make(domain, task, 3, 2, 0, None)
    action_shape = dummy_env.action_spec().shape
    obs_shape = dummy_env.observation_spec().shape

    # define architectures
    encoder = Encoder(obs_shape).cuda()
    if "drqv2_default" in algo:
        actor = ActorDefault(
            encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).cuda()
    else:
        actor = ActorSimple(
            encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).cuda()

    # load trained weights
    encoder.load(os.path.join(saved_weights_dir, "encoder/1000000.pth"))
    actor.load(os.path.join(saved_weights_dir, "actor/1000000.pth"))

    # run eval parallelly
    # with tqdm_joblib(total=test_episodes):
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_run_eval)(
            domain,
            task,
            distracting_mode,
            encoder,
            actor,
            stddev_schedule,
            kwargs_dict,
        )
        for _ in range(test_episodes)
    )

    return np.array(results).mean()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--n-jobs", default=4, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def main():

    print("=" * 65)
    args = arguments()
    now = datetime.datetime.now()
    start_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"start time: {start_time}")
    print()

    # get all tasks
    if not args.task:
        tasks = [
            p.path
            for p in os.scandir(f"{args.env_dir}")
            if os.path.isdir(p.path) and "extended" not in p.path
        ]
    else:
        tasks = [os.path.join(args.env_dir, args.task)]

    # loop through tasks
    for task in tasks:
        algos = [p.path for p in os.scandir(task) if os.path.isdir(p.path)]

        # loop through algorithms
        algo_mean = {}
        for algo in algos:
            print(f"task: {task}")
            print(f"algo: {algo}")
            runs = [p.path for p in os.scandir(algo) if os.path.isdir(p.path)]
            pbar3 = tqdm(runs)

            # loop through seeds and algo-variations
            runs_mean = {}
            for run in pbar3:
                pbar3.set_description(f"run: {run}")

                # actual eval on all modes
                mode_mean = {}
                pbar4 = tqdm(distracting_modes, leave=False)
                for distracting_mode in pbar4:
                    pbar4.set_description(f"mode: {distracting_mode}")

                    # calculate
                    if args.debug:
                        means = distracting_modes.index(distracting_mode)
                    else:
                        means = load_weights_and_eval_one_mode(
                            run, args.n_jobs, distracting_mode, algo,
                        )
                    mode_mean[distracting_mode] = means

                runs_mean[run] = mode_mean
            algo_mean[algo] = runs_mean

        os.makedirs("logs/ablation/", exist_ok=True)
        np.save(f"logs/ablation/{task.replace('/', '_')}.npy", algo_mean)

    print("-" * 40)
    end_time = datetime.datetime.now()
    end_time = end_time.strftime("%Y/%m/%d %H:%M:%S")
    print(f"execute time: {start_time}")
    print(f"end time: {end_time}")
    print()


if __name__ == "__main__":
    main()
