# common
import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf
import torch


def make_agent(obs_spec, action_spec, device, cfg):
    OmegaConf.update(cfg.algo.agent, "obs_dim", obs_spec.shape, merge=False)
    OmegaConf.update(
        cfg.algo.agent, "action_dim", action_spec.shape, merge=False
    )
    OmegaConf.update(cfg.algo.agent, "device", device, merge=False)
    return hydra.utils.instantiate(cfg.algo.agent)


def log_bias_evaluation(
    final_obs_list,
    final_act_list,
    final_mc_entropy_list,
    agent,
    device,
    mode,
    global_step,
):
    final_mc_entropy_list = np.array(final_mc_entropy_list).flatten()
    final_obs_list = np.array(final_obs_list)
    final_obs_list = final_obs_list.reshape((-1,) + final_obs_list.shape[2:])
    final_act_list = np.array(final_act_list)
    final_act_list = final_act_list.reshape((-1,) + final_act_list.shape[2:])

    with torch.no_grad():
        q_prediction = []
        for obs, act in zip(
            np.array_split(final_obs_list, 20),
            np.array_split(final_act_list, 20),
        ):
            obs_tensor = torch.Tensor(obs).to(device)
            acts_tensor = torch.Tensor(act).to(device)
            q_pred = (
                agent.get_ave_q_prediction_for_bias(obs_tensor, acts_tensor)
                .cpu()
                .numpy()
                .reshape(-1)
            )
            q_prediction.append(q_pred)
        q_prediction = np.concatenate(q_prediction)

    bias = q_prediction - final_mc_entropy_list
    bias_squared = bias**2
    final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
    final_mc_entropy_list_normalize_base = np.abs(
        final_mc_entropy_list_normalize_base
    )
    final_mc_entropy_list_normalize_base[
        final_mc_entropy_list_normalize_base < 10
    ] = 10

    normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
    normalized_bias_sqr_per_state = (
        bias_squared / final_mc_entropy_list_normalize_base
    )

    if mode is None:
        wandb.log(
            {
                "normal_bias_mean": np.mean(normalized_bias_per_state),
                "normal_bias_std": np.std(normalized_bias_per_state),
                "normal_bias_sq_mean": np.mean(normalized_bias_sqr_per_state),
                "normal_bias_sq_std": np.std(normalized_bias_sqr_per_state),
                "step": global_step,
            }
        )
    else:
        wandb.log(
            {
                f"{mode}_bias_mean": np.mean(normalized_bias_per_state),
                f"{mode}_bias_std": np.std(normalized_bias_per_state),
                f"{mode}_bias_sq_mean": np.mean(normalized_bias_sqr_per_state),
                f"{mode}_bias_sq_std": np.std(normalized_bias_sqr_per_state),
                "step": global_step,
            }
        )
