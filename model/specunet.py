# Ke Chen

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

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_emb(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.uniform_(layer.weight, -0.1, 0.1)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


def act(x, activation):
    if activation == 'relu':
        return F.relu_(x)

    elif activation == 'leaky_relu':
        return F.leaky_relu_(x, negative_slope=0.2)

    elif activation == 'swish':
        return x * torch.sigmoid(x)

    else:
        raise Exception('Incorrect activation!')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, activation, momentum, classes_num = 527):
        super(ConvBlock, self).__init__()

        self.activation = activation
        pad = size // 2

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(size, size), stride=(1, 1), 
                              dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(size, size), stride=(1, 1), 
                              dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        # change autotagging size
        # Mask Change
        # self.emb1 = nn.Linear(classes_num, out_channels, bias=True)
        # self.emb2 = nn.Linear(classes_num, out_channels, bias=True)

        self.init_weights()
        
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        # Mask Change
        # init_emb(self.emb1)
        # init_emb(self.emb2)

    # latent query embedded 
    def forward(self, x, condition = None):
        # Mask Change
        # c1 = self.emb1(condition)
        # c2 = self.emb2(condition)
        # x = act(self.bn1(self.conv1(x)), self.activation) + c1[:, :, None, None]
        # x = act(self.bn2(self.conv2(x)), self.activation) + c2[:, :, None, None]
        x = act(self.bn1(self.conv1(x)), self.activation)
        x = act(self.bn2(self.conv2(x)), self.activation)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum, classes_num = 527):
        super(EncoderBlock, self).__init__()
        size = 3 # Changeh here

        self.conv_block = ConvBlock(in_channels, out_channels, size, activation, momentum, classes_num)
        self.downsample = downsample

    def forward(self, x, condition = None):
        encoder = self.conv_block(x, condition)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum, classes_num = 527):
        super(DecoderBlock, self).__init__()
        size = 3 # Change here
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels, 
            out_channels=out_channels, kernel_size=(size, size), stride=stride, 
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=(1, 1))
        # Change padding here

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv_block2 = ConvBlock(out_channels * 2, out_channels, size, activation, momentum, classes_num)
        # change autotagging size
        # Mask Change
        # self.emb1 = nn.Linear(classes_num, out_channels, bias=True)

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn)
        # init_emb(self.emb1)

    def prune(self, x):
        """Prune the shape of x after transpose convolution.
        """
        x = x[:, :, 0 : - 1, 0 : - 1]
        return x

    def forward(self, input_tensor, concat_tensor, condition = None):
        # Mask Change
        # c1 = self.emb1(condition)
        # x = act(self.bn1(self.conv1(input_tensor)), self.activation) + c1[:, :, None, None]
        x = act(self.bn1(self.conv1(input_tensor)), self.activation)
        x = self.prune(x)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, condition)
        return x

# the main code for the music chorale separation
class MCS_SpecUNet(pl.LightningModule):
    '''
    Args:
    channels (int): the audio channel, default:1 (mono)
    config (module): the configuration module as in config.py
    at_model (module): the sound event detection system
    dataset (module): the dataset variable to control the randomness in each epoch (not affect in evaluation mode) 
    '''
    def __init__(self, channels, config, dataset, wav_output = False):
        super().__init__()

        # hyper parameters
        window_size = 2048
        hop_size = config.hop_samples
        center = True
        pad_mode = 'reflect'
        window = 'hann'
        activation = 'relu'
        momentum = 0.01
        self.check_flag = False
        self.config = config
        self.dataset = dataset
        self.wav_output = wav_output

        self.loss_func = get_loss_func(self.config.loss_type)

        self.downsample_ratio = 2 ** 6   # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

        self.istft = ISTFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlock(in_channels=channels, out_channels=32, 
            downsample=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.encoder_block2 = EncoderBlock(in_channels=32, out_channels=64, 
            downsample=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.encoder_block3 = EncoderBlock(in_channels=64, out_channels=128, 
            downsample=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.encoder_block4 = EncoderBlock(in_channels=128, out_channels=256, 
            downsample=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.encoder_block5 = EncoderBlock(in_channels=256, out_channels=512, 
            downsample=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.encoder_block6 = EncoderBlock(in_channels=512, out_channels=1024, 
            downsample=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.conv_block7 = ConvBlock(in_channels=1024, out_channels=2048, 
            size=3, activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.decoder_block1 = DecoderBlock(in_channels=2048, out_channels=1024, 
            stride=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.decoder_block2 = DecoderBlock(in_channels=1024, out_channels=512, 
            stride=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.decoder_block3 = DecoderBlock(in_channels=512, out_channels=256, 
            stride=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.decoder_block4 = DecoderBlock(in_channels=256, out_channels=128, 
            stride=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.decoder_block5 = DecoderBlock(in_channels=128, out_channels=64, 
            stride=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)
        self.decoder_block6 = DecoderBlock(in_channels=64, out_channels=32, 
            stride=(2, 2), activation=activation, momentum=momentum, classes_num = config.latent_dim)

        self.after_conv_block1 = ConvBlock(in_channels=32, out_channels=32, 
            size=3, activation=activation, momentum=momentum, classes_num = config.latent_dim)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=channels, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def spectrogram(self, input):
        (real, imag) = self.stft(input)
        return (real ** 2 + imag ** 2) ** 0.5

    def wav_to_spectrogram(self, input):
        """Waveform to spectrogram.
        Args:
          input: (batch_size, segment_samples, channels_num)
        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[2]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, :, channel]))

        output = torch.cat(sp_list, dim=1)
        return output


    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.
        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)
        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[2]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, :, channel])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(self.istft(spectrogram[:, channel : channel + 1, :, :] * cos, 
                spectrogram[:, channel : channel + 1, :, :] * sin, length))
        
        output = torch.stack(wav_list, dim=2)
        return output

    def forward(self, input, condition = None):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)
        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        sp = self.wav_to_spectrogram(input)    
        """(batch_size, channels_num, time_steps, freq_bins)"""

        # Batch normalization
        x = sp.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) \
            * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]     # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, condition)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool, condition)    # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool, condition)    # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool, condition)    # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool, condition)    # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool, condition)    # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, condition)    # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, condition)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, condition)    # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, condition)    # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, condition)   # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, condition)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, condition)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, condition)     # (bs, 32, T, F)
        x = self.after_conv2(x)             # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0 : origin_len, :]

        sp_out = torch.sigmoid(x) * sp

        # Spectrogram to wav
        length = input.shape[1]
        wav_out = self.spectrogram_to_wav(input, sp_out, length)

        output_dict = {"wav": wav_out, "sp": sp_out}
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
            # for mix
            # filename = os.path.join(test_output_path, self.config.dataset_name + "_" + audio_name + "_" + self.config.mix_name + ".wav")
            # sf.write(filename, whole_mixture, self.config.sample_rate)

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
        # print("Each Mean SDR:", mean_sdr)
        # print("Each Median SDR:", median_sdr)
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

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max",
            factor=0.65, patience=3,verbose=True
        )
        cop_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "mean_sdr"
            }
        } 

        return cop_dict
        # optimizer = optim.Adam(
        #     self.parameters(), lr = self.config.learning_rate, 
        #     betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
        # )
        # def lr_foo(epoch):       
        #     if epoch < 3:
        #         # warm up lr
        #         lr_scale = 1
        #     else:
        #         lr_scale = 0.5 ** (bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))

        #     return lr_scale
        # scheduler = optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lr_foo
        # )

        # return [optimizer], [scheduler]

