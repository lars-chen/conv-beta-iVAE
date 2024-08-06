import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


##
def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(
            channel_in,
            (channel_out // 2) + channel_out,
            kernel_size,
            2,
            kernel_size // 2,
        )
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(
            channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2
        )

        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(
        self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"
    ):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(
            channel_in,
            (channel_in // 2) + channel_out,
            kernel_size,
            1,
            kernel_size // 2,
        )
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(
            channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2
        )

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        first_out = (
            channel_in // 2
            if channel_in == channel_out
            else (channel_in // 2) + channel_out
        )
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(
            channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2
        )
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))

        x_cat = self.conv1(x)
        x = x_cat[:, : self.bttl_nk]

        # If channel_in == channel_out we do a simple identity skip
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(
        self,
        channels,
        ch=64,
        image_size=128,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(Encoder, self).__init__()
        fm_size = image_size // (2 ** len(blocks))
        self.intermediate_shape = latent_channels * fm_size * fm_size
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]

        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:
                # Add an additional non down-sampling block before down-sampling
                self.layer_blocks.append(
                    ResBlock(w_in * ch, w_in * ch, norm_type=norm_type)
                )

            self.layer_blocks.append(
                ResDown(w_in * ch, w_out * ch, norm_type=norm_type)
            )

        for _ in range(num_res_blocks):
            self.layer_blocks.append(
                ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type)
            )

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

        self.fc_mu = nn.Linear(self.intermediate_shape, latent_channels)
        self.fc_log_var = nn.Linear(self.intermediate_shape, latent_channels)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, x, t, sample=False):
        x_channels = x[:, :, None, None].expand(
            x.shape[0], x.shape[1], y.shape[2], y.shape[3]
        )
        t_channel = t[:, None, None, None].expand(x.shape[0], 1, y.shape[2], y.shape[3])
        encoder_input = torch.cat((y, x_channels, t_channel), dim=1)

        y = self.conv_in(encoder_input)

        for block in self.layer_blocks:
            y = block(y)
        y = self.act_fnc(y)

        mu = self.conv_mu(y)
        mu = self.fc_mu(mu.view(-1, self.intermediate_shape))

        log_var = self.conv_log_var(y)
        log_var = self.fc_log_var(log_var.view(-1, self.intermediate_shape))

        if self.training or sample:
            z = self.sample(mu, log_var)
        else:
            z = mu

        return z, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(
        self,
        channels,
        ch=64,
        blocks=(1, 2, 4, 8),
        image_size=128,
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]

        self.fm_size = image_size // 2 ** (len(blocks))
        self.intermediate_shape = latent_channels * self.fm_size * self.fm_size
        self.latent_channels = latent_channels

        self.fc_in = nn.Linear(latent_channels + 1, self.intermediate_shape)
        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)

        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(
                ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type)
            )

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(
                    ResBlock(w_out * ch, w_out * ch, norm_type=norm_type)
                )

        # self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.conv_out_mu = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.conv_out_logvar = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, y, t):
        # t_channel = t[:, None, None, None].expand(y.shape[0], 1, y.shape[2], y.shape[3])
        t = t.unsqueeze(1)
        y = torch.cat((y, t), dim=1)
        y = self.fc_in(y)
        y = self.conv_in(y.view(-1, self.latent_channels, self.fm_size, self.fm_size))

        for block in self.layer_blocks:
            y = block(y)
        y = self.act_fnc(y)

        return torch.tanh(self.conv_out_mu(y))


# , torch.tanh(self.conv_out_logvar(y)) torch.tanh(self.conv_out(y))


class LearnedPrior(nn.Module):
    def __init__(self, latent_dim=256, label_dim=11):
        super(LearnedPrior, self).__init__()
        self.latent_dim = latent_dim

        self.prior_nn = nn.Sequential(
            nn.Linear(label_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )

        self.prior_mu = nn.Linear(256, latent_dim)
        self.prior_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        prior_params = self.prior_nn(x.type(torch.float32))
        prior_mu = self.prior_mu(prior_params)
        prior_logvar = self.prior_logvar(prior_params)
        return prior_mu, prior_logvar


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """

    def __init__(
        self,
        image_size=128,
        label_dim=11,
        channel_in=3,
        ch=64,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(VAE, self).__init__()

        self.encoder = Encoder(
            channel_in + label_dim + 1,
            ch=ch,
            image_size=image_size,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )
        self.decoder = Decoder(
            channel_in,
            ch=ch,
            image_size=image_size,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.prior = LearnedPrior(latent_channels, label_dim=label_dim)

    def forward(self, y, x, t):
        encoding, mu, log_var = self.encoder(y, x, t)
        recon_img = self.decoder(encoding, t)

        prior_params = self.prior(x)
        return (recon_img, mu, log_var, prior_params)
