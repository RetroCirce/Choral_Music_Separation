# Ke Chen
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn.abn import InPlaceABNSync
from torchlibrosa.stft import ISTFT, STFT, magphase
import pytorch_lightning as pl
import torch.distributed as dist
import torch.optim as optim
from functools import partial

import soundfile as sf
from utils import np_to_pytorch, evaluate_sdr, get_lr_lambda
from losses import get_loss_func



def init_layer(layer: nn.Module):
    r"""Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module):
    r"""Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)

class Base:
    def __init__(self):
        r"""Base function for extracting spectrogram, cos, and sin, etc."""
        pass

    def spectrogram(self, input: torch.Tensor, eps: float = 0.0):
        r"""Calculate spectrogram.
        Args:
            input: (batch_size, segments_num)
            eps: float
        Returns:
            spectrogram: (batch_size, time_steps, freq_bins)
        """
        (real, imag) = self.stft(input)
        return torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5

    def spectrogram_phase(
        self, input: torch.Tensor, eps: float = 0.0
    ):
        r"""Calculate the magnitude, cos, and sin of the STFT of input.
        Args:
            input: (batch_size, segments_num)
            eps: float
        Returns:
            mag: (batch_size, time_steps, freq_bins)
            cos: (batch_size, time_steps, freq_bins)
            sin: (batch_size, time_steps, freq_bins)
        """
        (real, imag) = self.stft(input)
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(
        self, input: torch.Tensor, eps: float = 1e-10
    ):
        r"""Convert waveforms to magnitude, cos, and sin of STFT.
        Args:
            input: (batch_size, channels_num, segment_samples)
            eps: float
        Outputs:
            mag: (batch_size, channels_num, time_steps, freq_bins)
            cos: (batch_size, channels_num, time_steps, freq_bins)
            sin: (batch_size, channels_num, time_steps, freq_bins)
        """
        batch_size, channels_num, segment_samples = input.shape

        # Reshape input with shapes of (n, segments_num) to meet the
        # requirements of the stft function.
        x = input.reshape(batch_size * channels_num, segment_samples)

        mag, cos, sin = self.spectrogram_phase(x, eps=eps)
        # mag, cos, sin: (batch_size * channels_num, 1, time_steps, freq_bins)

        _, _, time_steps, freq_bins = mag.shape
        mag = mag.reshape(batch_size, channels_num, time_steps, freq_bins)
        cos = cos.reshape(batch_size, channels_num, time_steps, freq_bins)
        sin = sin.reshape(batch_size, channels_num, time_steps, freq_bins)

        return mag, cos, sin

    def wav_to_spectrogram(
        self, input: torch.Tensor, eps: float = 1e-10
    ):

        mag, cos, sin = self.wav_to_spectrogram_phase(input, eps)
        return mag

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        self.activation = activation
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        # ABN is not used for bn1 because we found using abn1 will degrade performance.
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)

        self.abn2 = InPlaceABNSync(
            num_features=out_channels, momentum=momentum, activation='leaky_relu'
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(self.abn2(x))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


class EncoderBlockRes4B(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, downsample, activation, momentum
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes4B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, activation, momentum
        )
        self.conv_block2 = ConvBlockRes(
            out_channels, out_channels, kernel_size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, kernel_size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, kernel_size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes4B(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, upsample, activation, momentum
    ):
        r"""Decoder block, contains 1 transpose convolutional and 8 convolutional layers."""
        super(DecoderBlockRes4B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, kernel_size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, kernel_size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, kernel_size, activation, momentum
        )
        self.conv_block5 = ConvBlockRes(
            out_channels, out_channels, kernel_size, activation, momentum
        )

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, input_tensor, concat_tensor):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x

class MCS_DPIResUNet(pl.LightningModule, Base):
    def __init__(self, channels, config, dataset, wav_output = False):
        super(MCS_DPIResUNet, self).__init__()

        self.input_channels = channels
        self.target_sources_num = 1 # target_sources_num = 1

        self.config = config
        self.dataset = dataset
        self.wav_output = wav_output
        self.check_flag = False

        window_size = 2048
        hop_size = config.hop_samples
        center = True
        pad_mode = 'reflect'
        window = 'hann'
        activation = 'leaky_relu'
        momentum = 0.01

        self.loss_func = get_loss_func(self.config.loss_type)

        self.lr_lambda = partial(
            get_lr_lambda, 
            warm_up_steps = self.config.resunet_warmup_steps, 
            reduce_lr_steps = self.config.resunet_reduce_lr_steps
        )
        self.subbands_num = 1
        

        assert (
            self.subbands_num == 1
        ), "Using subbands_num > 1 on spectrogram \
            will lead to unexpected performance sometimes. Suggest to use \
            subband method on waveform."

        # Downsample rate along the time axis.
        self.K = 4  # outputs: |M|, cos∠M, sin∠M, Q
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes4B(
            in_channels=channels * self.subbands_num,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlockRes4B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlockRes4B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlockRes4B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlockRes4B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlockRes4B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7a = EncoderBlockRes4B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7b = EncoderBlockRes4B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7c = EncoderBlockRes4B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7d = EncoderBlockRes4B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlockRes4B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlockRes4B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlockRes4B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlockRes4B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlockRes4B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlockRes4B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = EncoderBlockRes4B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(1, 1),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=self.target_sources_num
            * channels
            * self.K
            * self.subbands_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ):
        r"""Convert feature maps to waveform.
        Args:
            input_tensor: (batch_size, target_sources_num * input_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
            sin_in: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
            cos_in: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
        Outputs:
            waveform: (batch_size, target_sources_num * input_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.input_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, input_channles, K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        linear_mag = x[:, :, :, 3, :, :]
        # mask_cos, mask_sin: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Reformat shape to (n, 1, time_steps, freq_bins) for ISTFT.
        shape = (
            batch_size * self.target_sources_num * self.input_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * input_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.input_channels, audio_length
        )
        # (batch_size, target_sources_num * input_channels, segments_num)

        return waveform

    def forward(self, input):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)
        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
        }
        """
        mixtures = input.permute(0,2,1).contiguous()
        # (batch_size, input_channels, segment_samples)

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        # mag, cos_in, sin_in: (batch_size, input_channels, time_steps, freq_bins)

        # Batch normalize on individual frequency bins.
        x = mag.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # x: (batch_size, input_channels, time_steps, freq_bins)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # (batch_size, channels, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 1025 -> 1024.
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        if self.subbands_num > 1:
            x = self.subband.analysis(x)
            # (bs, input_channels, T, F'), where F' = F // subbands_num

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool
        )  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool
        )  # x5_pool: (bs, 384, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool
        )  # x6_pool: (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7a(x6_pool)  # (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7b(x_center)  # (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7c(x_center)  # (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7d(x_center)  # (bs, 384, T / 32, F / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        (x, _) = self.after_conv_block1(x12)  # (bs, 32, T, F)

        x = self.after_conv2(x)  # (bs, channels * 3, T, F)
        # (batch_size, target_sources_num * input_channles * self.K * subbands_num, T, F')

        if self.subbands_num > 1:
            x = self.subband.synthesis(x)
            # (batch_size, target_sources_num * input_channles * self.K, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 1024 -> 1025.

        x = x[:, :, 0:origin_len, :]
        # (batch_size, target_sources_num * input_channles * self.K, T, F)

        audio_length = mixtures.shape[2]

        separated_audio = self.feature_maps_to_wav(x, mag, sin_in, cos_in, audio_length)
        # separated_audio: (batch_size, target_sources_num * input_channels, segments_num)
        separated_audio = separated_audio.permute(0,2,1).contiguous()
        output_dict = {'wav': separated_audio}

        return output_dict

    def training_step(self, batch, batch_idx):
        self.train()
        self.device_type = next(self.parameters()).device
        if not self.check_flag:
            self.check_flag = True
        mixtures = np_to_pytorch(np.array(batch["mixture"])[:, :, None], self.device_type)
        sources = np_to_pytorch(np.array(batch["source"])[:, :, None], self.device_type)
        if len(mixtures) > 0:
            # train
            batch_output_dict = self(mixtures)
            loss = self.loss_func(batch_output_dict["wav"], sources)
            return loss
        else:
            return None

    def training_epoch_end(self, outputs):
        self.check_flag = False
        self.dataset.generate_queue()

    def validation_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        audio_len = len(batch["mixture"][0])
        sample_len = self.config.segment_frames * self.config.hop_samples
        audio_len = (audio_len // sample_len) * sample_len

        whole_mixture = batch["mixture"][0][:audio_len]
        whole_source = batch["source"][0][:audio_len]

        split_mixtures = np.array(np.split(whole_mixture, audio_len // sample_len))
        split_sources = np.array(np.split(whole_source, audio_len // sample_len))
        sdr = []
        whole_pred = []
        # batch feed
        batch_size = self.config.batch_size // torch.cuda.device_count()
        for i in range(0, len(split_mixtures), batch_size):
            mixtures = np_to_pytorch(split_mixtures[i:i + batch_size].copy()[:, :, None], self.device_type)
            sources = np_to_pytorch(split_sources[i:i + batch_size].copy()[:, :, None], self.device_type)
            if len(mixtures) > 0:
                # validate
                batch_output_dict = self(mixtures)
                preds = batch_output_dict["wav"]
                temp_sdr = evaluate_sdr(
                    ref = sources.data.cpu().numpy(), 
                    est = preds.data.cpu().numpy(),
                    class_ids = np.array([1] * len(sources)),
                    mix_type = "mixture"
                )
                if self.wav_output:
                    preds = preds.view(-1).data.cpu().numpy().tolist()
                    whole_pred += preds
                sdr += temp_sdr
        if self.wav_output:
            test_output_path = os.path.join(self.config.workspace, self.config.test_output)
            audio_name = batch["audio_name"][0]
            filename = os.path.join(test_output_path, self.config.dataset_name + "_" + audio_name + "_" + self.config.sep_track + ".wav")
            sf.write(filename, whole_pred, self.config.sample_rate)
        return {"sdr": sdr}

    def validation_epoch_end(self, validation_step_outputs):
        self.device_type = next(self.parameters()).device
        mean_sdr = []
        median_sdr = []
        for d in validation_step_outputs:
            mean_sdr.append(np.mean([dd[0][0] for dd in d["sdr"]]))
            median_sdr.append(np.median([dd[0][0] for dd in d["sdr"]]))
        mean_sdr = np.array(mean_sdr)
        median_sdr = np.array(median_sdr)
        # ddp 
        if torch.cuda.device_count() == 1:
            self.print("--------Single GPU----------")
            metric_mean_sdr = np.mean(mean_sdr)
            metric_median_sdr = np.median(median_sdr)
            self.log("mean_sdr", metric_mean_sdr, on_epoch = True, prog_bar=True, sync_dist=True)
            self.log("median_sdr", metric_median_sdr, on_epoch = True, prog_bar=True, sync_dist=True)
            self.print("Median SDR:", metric_median_sdr,"| Mean SDR:", metric_mean_sdr)
        else:
            mean_sdr = np_to_pytorch(mean_sdr, self.device_type)
            median_sdr = np_to_pytorch(median_sdr, self.device_type)
            gather_mean_sdr = [torch.zeros_like(mean_sdr) for _ in range(dist.get_world_size())]
            gather_median_sdr = [torch.zeros_like(median_sdr) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(gather_mean_sdr, mean_sdr)
            dist.all_gather(gather_median_sdr, median_sdr)
            metric_mean_sdr = 0.0
            metric_median_sdr = 0.0
            if dist.get_rank() == 0:
                gather_mean_sdr = torch.cat(gather_mean_sdr, dim = 0).cpu().numpy()
                gather_median_sdr = torch.cat(gather_median_sdr, dim = 0).cpu().numpy()
                print(gather_mean_sdr.shape)
                print(gather_median_sdr.shape)
                metric_mean_sdr = np.mean(gather_mean_sdr)
                metric_median_sdr = np.median(gather_median_sdr)
            self.log("mean_sdr", metric_mean_sdr * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            self.log("median_sdr", metric_median_sdr * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            self.print("Median SDR:", metric_median_sdr,"| Mean SDR:", metric_mean_sdr)
            dist.barrier()
        
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        self.validation_epoch_end(test_step_outputs)             

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr = self.config.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
        )

        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]