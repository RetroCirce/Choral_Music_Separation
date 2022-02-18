# Ke Chen

from distutils.command.config import config
import numpy as np
import os
import math
import bisect
import pickle
import soundfile as sf

from utils import np_to_pytorch, evaluate_sdr
from losses import get_loss_func

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchlibrosa.stft import STFT, ISTFT, magphase
import pytorch_lightning as pl
import torch.distributed as dist

from torch.autograd import Variable

EPS = 1e-8

def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().to(signal.device)  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result



class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        # Components
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3) # [M, C, K, N]
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source




class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm,
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                     pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else: # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [TemporalBlock(B, H, P, stride=1,
                                         padding=padding,
                                         dilation=dilation,
                                         norm_type=norm_type,
                                         causal=causal)]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C*N, 1, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm,
                                     bottleneck_conv1x1,
                                     temporal_conv_net,
                                     mask_conv1x1)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        score = score.view(M, self.C, N, K) # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class MCS_ConvTasNet(pl.LightningModule):
    def __init__(self, channels, config, dataset):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self.config = config
        self.check_flag = False
        self.N = config.tasnet_enc_dim
        self.L = config.tasnet_filter_length
        self.B = config.tasnet_feature_dim
        self.H = config.tasnet_hidden_dim
        self.P = config.tasnet_kernel
        self.X = config.tasnet_layer
        self.R = config.tasnet_stack
        self.C = channels
        self.norm_type = "gLN"
        self.causal = config.tasnet_causal
        self.mask_nonlinear = 'relu'
        self.dataset = dataset
        self.loss_func = get_loss_func(config.loss_type)

        # Components
        self.encoder = Encoder(self.L, self.N)
        self.separator = TemporalConvNet(self.N, self.B, self.H, self.P, self.X, self.R, self.C, self.norm_type, self.causal, self.mask_nonlinear)
        self.decoder = Decoder(self.N, self.L)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source
    
    def training_step(self, batch, batch_idx):
        self.train()
        self.device_type = next(self.parameters()).device
        if not self.check_flag:
            self.check_flag = True
        mixtures = np_to_pytorch(np.array(batch["mixture"]), self.device_type)
        sources = np_to_pytorch(np.array(batch["source"])[:, None, :], self.device_type)
        if len(mixtures) > 0:
            # train
            batch_output_dict = self(mixtures)
            loss = self.loss_func(batch_output_dict, sources)
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
        # batch feed
        batch_size = self.config.batch_size // torch.cuda.device_count()
        for i in range(0, len(split_mixtures), batch_size):
            mixtures = np_to_pytorch(split_mixtures[i:i + batch_size].copy(), self.device_type)
            sources = np_to_pytorch(split_sources[i:i + batch_size].copy()[:, :, None], self.device_type)

            if len(mixtures) > 0:
                # validate
                batch_output_dict = self(mixtures) # B C T
                preds = torch.permute(batch_output_dict, (0,2,1)) # B T C
                temp_sdr = evaluate_sdr(
                    ref = sources.data.cpu().numpy(), 
                    est = preds.data.cpu().numpy(),
                    class_ids = np.array([1] * len(sources)),
                    mix_type = "mixture"
                )
                sdr += temp_sdr
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
        def lr_foo(epoch):       
            if epoch < 3:
                # warm up lr
                lr_scale = 1
            else:
                lr_scale = 0.5 ** (bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
            return lr_scale
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )

        return [optimizer], [scheduler]

