import torch.nn.functional as F
import numpy as np
import yaml
import os
import shutil


##
def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def count_parameters(model):
    """returns the total number of model parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def read_config(path=None):

    if path == None:
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
    else:
        with open(path, "r") as file:
            config = yaml.safe_load(file)

    label_idxs = np.array(list(config["model_labels"].values()), dtype=int)  # [
    #     config["labels"]["Young"],
    #     config["labels"]["Smiling"],
    #     config["labels"]["Mouth_Slightly_Open"],
    #     config["labels"]["No_Beard"],
    #     config["labels"]["Bald"],
    #     config["labels"]["5_o_Clock_Shadow"],  # config["labels"]["Pale_Skin"]
    #     config["labels"]["Mustache"],
    #     config["labels"]["Heavy_Makeup"],
    #     config["labels"]["Gray_Hair"],
    #     config["labels"]["Bangs"],
    #     config["labels"]["High_Cheekbones"],
    # ]
    t_idx = list(config["treatment"].values())[0]  # config["labels"]["Male"]
    learning_rate = config["lr"]
    norm_type = config["norm_type"]
    nepochs = config["nepoch"]
    image_size = config["image_size"]
    ch_multi = config["ch_multi"]
    num_res_blocks = config["num_res_blocks"]
    gpu_index = config["gpu_index"]
    latent_channels = config["latent_channels"]
    save_interval = config["save_interval"]
    kl_scale = config["kl_scale"]
    deep_model = config["deep_model"]
    block_widths = np.array(config["block_widths"].split(","), dtype=int)
    save_dir = config["save_dir"]
    return (
        label_idxs,
        t_idx,
        norm_type,
        kl_scale,
        learning_rate,
        nepochs,
        image_size,
        ch_multi,
        num_res_blocks,
        gpu_index,
        latent_channels,
        deep_model,
        save_interval,
        block_widths,
        save_dir,
    )


def kl_divergence(mu_p, sigma_p, mu_q, sigma_q):
    """KL divergence between two multivarariate Gaussian distributions KLD(p || q)?

    Args:
        mu_p (_type_): _description_
        sigma_p (_type_): _description_
        mu_q (_type_): _description_
        sigma_q (_type_): _description_

    Returns:
        _type_: _description_
    """
    kl_divergence = (
        (mu_q - mu_p).T @ np.linalg.inv(sigma_q) @ (mu_q - mu_p)
        + np.trace(np.linalg.inv(sigma_q) @ sigma_p)
        - np.log(np.linalg.det(sigma_p) / np.linalg.det(sigma_q))
    )
    return kl_divergence


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def create_save_folder(copy_config=True):
    if not os.path.isdir(save_dir + "/Runs"):
        save_dir += "/Runs"
        os.makedirs(save_dir)
    runs_list = os.listdir(save_dir + "/Runs")
    if runs_list == []:
        os.makedirs(save_dir + "/Runs" + "/Run_1")
        save_dir += "/Runs" + "/Run_1"
    else:
        new_run = "/Run_" + str(
            np.array([int(run.split("_")[-1]) for run in runs_list]).max() + 1
        )
        save_dir += "/Runs" + new_run
        os.makedirs(save_dir)

    if copy_config:
        shutil.copyfile("config.yml", save_dir + "/config.yml")
