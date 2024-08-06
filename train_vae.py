import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch import nn
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from torch.distributions import kl_divergence, Independent, Normal


import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

# project modules
import helpers as hf

from RES_VAE_Dynamic import VAE

##
(
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
) = hf.read_config()


parser = argparse.ArgumentParser(description="Training Params")
# string args

parser.add_argument(
    "--batch_size", "-bs", help="Training batch size", type=int, default=32
)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(gpu_index if use_cuda else "cpu")
print("")

# Create dataloaders
print("-Target Image Size %d" % image_size)
celeb_transform = transforms.Compose(
    [
        transforms.CenterCrop(150),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

data_dir = "../../../../../groups/kempter/chen/data"
# download dataset
train_dataset = Datasets.CelebA(
    data_dir, transform=celeb_transform, download=False, split="train"
)
train_dataset = Subset(
    dataset=train_dataset, indices=np.arange(16, 162770, 1)
)  # 162770
test_dataset = Datasets.CelebA(
    data_dir, transform=celeb_transform, download=False, split="valid"
)

# create train and test dataloaders
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, num_workers=16, shuffle=False
)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)
test_images, labels = next(dataiter)
test_labels = labels[:, label_idxs]
test_treatment = labels[:, t_idx]

# Create AE network.
vae_net = VAE(
    channel_in=test_images.shape[1],
    ch=ch_multi,
    blocks=block_widths,
    latent_channels=latent_channels,
    num_res_blocks=num_res_blocks,
    norm_type=norm_type,
    deep_model=deep_model,
    image_size=image_size,
    label_dim=len(label_idxs),
).to(device)

# Setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=learning_rate)

# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

if norm_type == "bn":
    print("-Using BatchNorm")
elif norm_type == "gn":
    print("-Using GroupNorm")
else:
    ValueError("norm_type must be bn or gn")

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in vae_net.parameters():
    num_model_params += param.flatten().shape[0]

print(
    "-This Model Has %d (Approximately %d Million) Parameters!"
    % (num_model_params, num_model_params // 1e6)
)
fm_size = image_size // (2 ** len(block_widths))
print("-The Latent Space Size Is %dx%dx%d!" % (latent_channels, fm_size, fm_size))

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

shutil.copyfile("config.yml", save_dir + "/config.yml")
save_file_name = "model" + "_" + str(image_size)

print("Starting from scratch")
start_epoch = 0
# Loss and metrics logger
data_logger = defaultdict(lambda: [])

# Start training loop
for epoch in trange(start_epoch, nepochs, leave=False):
    vae_net.train()
    for i, (images, labels) in enumerate(tqdm(train_loader, leave=False)):

        current_iter = i + epoch * len(train_loader)
        images = images.to(device)
        bs, c, h, w = images.shape

        xs = labels[:, label_idxs].to(device)
        ts = labels[:, t_idx].to(device)

        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            recon_img, mu, log_var, prior_params = vae_net(images, xs, ts)

            prior_mu = prior_params[0]
            prior_log_var = prior_params[0]

            prior = Independent(Normal(prior_mu, F.softplus(prior_log_var)), 1)
            q_approx = Independent(Normal(mu, F.softplus(log_var)), 1)

            # kl_loss = kl_divergence(q_approx, prior).mean()
            kl_loss = hf.kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, images)

            elbo = kl_scale * kl_loss + mse_loss
            loss = elbo

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()

        # Log losses and other metrics for evaluation!
        data_logger["mse"].append(mse_loss.mean().item())
        data_logger["kl"].append(kl_loss.mean().item())
        data_logger["mu"].append(mu.mean().item())
        data_logger["mu_var"].append(mu.var().item())
        data_logger["log_var"].append(log_var.mean().item())
        data_logger["log_var_var"].append(log_var.var().item())

        # Save results and a checkpoint at regular intervals
        if ((current_iter + 1) % save_interval == 0) or current_iter == 128:
            os.makedirs(save_dir + f"/epoch{epoch}_step_{current_iter}")

            # In eval mode the model will use mu as the encoding instead of sampling from the distribution
            vae_net.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Save an female example from testing and log a test loss
                    (female_img, mu, log_var, _) = vae_net(
                        test_images.to(device),
                        test_labels.to(device),
                        torch.zeros_like(test_treatment).to(device),
                    )

                    # Save an male example from testing and log a test loss
                    male_img, mu, log_var, _ = vae_net(
                        test_images.to(device),
                        test_labels.to(device),
                        torch.ones_like(test_treatment).to(device),
                    )

                    img_cat = torch.cat(
                        (test_images, female_img.cpu(), male_img.cpu()), 2
                    ).float()

                    vutils.save_image(
                        img_cat,
                        "%s/%s/test.png"
                        % (
                            save_dir,
                            f"epoch{epoch}_step_{current_iter}",
                        ),
                        normalize=True,
                    )

                    data_logger["test_mse_loss"].append(
                        F.mse_loss(recon_img, test_images.to(device)).item()
                    )

                # Save a checkpoint
                if epoch > 0:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "data_logger": dict(data_logger),
                            "model_state_dict": vae_net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        save_dir
                        + f"/epoch{epoch}_step_{current_iter}/"
                        + save_file_name
                        + ".pt",
                    )

                # Set the model back into training mode!!
                vae_net.train()
