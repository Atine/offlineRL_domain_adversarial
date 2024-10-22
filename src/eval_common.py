# common
import numpy as np
import os
from matplotlib import pyplot as plt
import wandb
import torch

# own
from src import dmc


def run_eval(
    domain,
    task,
    frame_stack,
    action_repeat,
    seed,
    distracting_mode,
    agent,
    gamma,
    alpha,
    video_recorder,
    save_video,
    curr_step,
    n_mc_cutoff=150,
    kwargs_dict=None,
):
    step = 0
    total_reward = 0
    ep_logprob_list = []
    ep_reward_list = []
    ep_obs_list = []
    ep_act_list = []

    env = dmc.make(
        domain,
        task,
        frame_stack,
        action_repeat,
        seed + 100,
        distracting_mode,
        kwargs_dict=kwargs_dict,
    )
    time_step = env.reset()

    # only record video at test episode == 0
    video_recorder.init(env, enabled=(save_video))

    while not time_step.last():
        eval_obs = time_step.observation
        eval_obs = eval_obs / 255.0
        with torch.no_grad():
            action, logprob_a_tilda = agent.act_with_logprob(
                eval_obs, curr_step, eval_mode=True
            )

        ep_obs_list.append(eval_obs)
        ep_act_list.append(action)
        time_step = env.step(action)
        video_recorder.record(env)
        total_reward += time_step.reward
        ep_reward_list.append(time_step.reward)
        ep_logprob_list.append(logprob_a_tilda.cpu().item())
        step += 1

    if distracting_mode is None:
        # for normal env
        video_recorder.save(f"normal_{curr_step}_" f"{total_reward}.mp4")
    else:
        # for distracting envs
        video_recorder.save(
            f"{distracting_mode}_{curr_step}_" f"{total_reward}.mp4"
        )

    # calc bias
    discounted_return_list = np.zeros(step)
    discounted_return_with_entropy_list = np.zeros(step)
    for i_step in range(step - 1, -1, -1):
        # backwards compute discounted return and with entropy
        # for all s-a visited
        if i_step == step - 1:
            discounted_return_list[i_step] = ep_reward_list[i_step]
            discounted_return_with_entropy_list[i_step] = ep_reward_list[
                i_step
            ]
        else:
            discounted_return_list[i_step] = (
                ep_reward_list[i_step]
                + gamma * discounted_return_list[i_step + 1]
            )
            discounted_return_with_entropy_list[i_step] = ep_reward_list[
                i_step
            ] + gamma * (
                discounted_return_with_entropy_list[i_step + 1]
                - alpha * ep_logprob_list[i_step + 1]
            )

    return (
        total_reward,
        discounted_return_with_entropy_list[:n_mc_cutoff],
        ep_obs_list[:n_mc_cutoff],
        ep_act_list[:n_mc_cutoff],
        step,
    )


def log_bellman(
    agent, replay_buffer, replay_buffer_dis, global_step, work_dir, cfg
):
    # log bellman distribution historgram
    error = agent.calculate_bellman_errors(replay_buffer, global_step)
    error_dis = agent.calculate_bellman_errors(replay_buffer_dis, global_step)

    # save bellman errors
    savename = f"{work_dir}/bellmanerrors/" f"step{int(global_step):>07}.npy"
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    with open(savename, "wb") as f:
        np.save(f, error)
        np.save(f, error_dis)

    # plot bellman histogram
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(error, bins=64, alpha=0.5, label="normal")
    ax.hist(error_dis, bins=64, alpha=0.5, label="dis")
    ax.legend(loc="upper left")
    ax.set_title(f"{cfg.experiment} - seed {cfg.seed}")
    savename = f"{work_dir}/histogram/" f"step{int(global_step):>07}.jpg"
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    plt.savefig(savename, bbox_inches="tight")
    plt.close()
    if not cfg.debug:
        wandb.log(
            {
                "step": global_step,
                "histogram": wandb.Image(savename),
            }
        )
