import os
import pprint
import numpy as np
import pandas as pd



modes = [
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



dfs = []
for p in os.scandir(
    "./logs/ablation/*.npy"
):

    npy_fp = p.path
    results_dict = np.load(npy_fp, allow_pickle=True).item()

    df1 = pd.DataFrame(columns=["task", "algo", "additional", "seed"])
    df2 = pd.DataFrame(columns=modes)

    # loop through tasks
    for k, v in results_dict.items():

        # loop through algos
        for k2, v2 in v.items():
            task = k.split("cheetah_run/")[1]
            algo = os.path.basename(task)
            task = os.path.dirname(task)
            seed = os.path.basename(k2).split("seed")[1].split("_")[0]
            runs_additional = "_".join(
                os.path.basename(k2).split("seed")[1].split("_")[3:]
            )
            # runs_base = os.path.basename(k2).split(f"_{runs_additional}")[0]
            # runs_base = algo
            df1.loc[len(df1)] = [task, algo, runs_additional, seed]
            df2.loc[len(df2)] = v2
            # print(task, algo, runs_additional, seed, v2)

            # loop though difficulties
            # for k3, v3 in v2.items():
            #     print()


    df = pd.concat([df1, df2], axis=1)
    dfs.append(df)
