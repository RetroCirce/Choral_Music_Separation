# Ke Chen

from pickletools import optimize
from sched import scheduler
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from utils import np_to_pytorch, evaluate_sdr
from losses import get_loss_func
import torch.distributed as dist
import torch.optim as optim



def centre_crop(x, target):
    '''
    Center-crop 3-dim. input tensor along last axis so it fits the target tensor shape
    :param x: Input tensor
    :param target: Shape of this tensor will be used as target shape
    :return: Cropped input tensor
    '''
    if x is None:
        return None
    if target is None:
        return x

    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]
    assert (diff % 2 == 0)
    crop = diff // 2

    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop:-crop].contiguous()


class Resample1d(nn.Module):
    def __init__(self, channels, kernel_size, stride, transpose=False, padding="reflect", trainable=False):
        '''
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialisation
        '''
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert(kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)
        assert(padding == "reflect" or padding == "valid")

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)), requires_grad=trainable)

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size-1)//2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:,:,diff_steps//2:-diff_steps//2]
        else:
            assert(input_size % self.stride == 1)
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out

    def get_output_size(self, input_size):
        '''
        Returns the output dimensionality (number of timesteps) for a given input size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''
        assert(input_size > 1)
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return ((input_size - 1) * self.stride + 1)
        else:
            assert(input_size % self.stride == 1) # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        '''
        Returns the input dimensionality (number of timesteps) for a given output size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''

        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size

def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
    # Sinc lowpass filter
    # Build sinc kernel
    assert(kernel_size % 2 == 1)
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M//2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M//2)) / (i - M//2)) * \
                    (0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M))

    filter = filter / np.sum(filter)
    return filter

class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        if self.transpose:
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size-1)
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "gn":
            assert(n_outputs % NORM_CHANNELS == 0)
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.conv_type == "gn" or self.conv_type == "bn":
            out = F.relu(self.norm((self.filter(x))))
        else: # Add your own variations here with elifs conditioned on "conv_type" parameter!
            assert(self.conv_type == "normal")
            out = F.leaky_relu(self.filter(x))
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            assert(input_size > 1)
            curr_size = (input_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = input_size

        # Conv
        curr_size = curr_size - self.kernel_size + 1 # o = i + p - k + 1
        assert (curr_size > 0)

        # Strided conv/decimation
        if not self.transpose:
            assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class MCS_Waveunet(pl.LightningModule):
    def __init__(self, channels, config, dataset, wav_output = False):
        super(MCS_Waveunet, self).__init__()

        self.dataset = dataset
        self.config = config
        self.check_flag = False
        self.num_levels = len(config.waveunet_features)
        self.strides = config.waveunet_stride
        self.kernel_size = config.waveunet_kernel
        self.num_inputs = channels
        self.num_outputs = channels
        self.depth = config.waveunet_depth
        self.instruments = ["ALL"]
        self.separate = False
        self.init_opt = False
        res = config.waveunet_res
        conv_type = config.waveunet_convtype
        # Only odd filter kernels allowed
        assert(self.kernel_size % 2 == 1)

        num_channels = config.waveunet_features
        self.waveunets = nn.ModuleDict()
        model_list = ["ALL"]

        self.loss_func = get_loss_func(config.loss_type)

        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = self.num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], self.kernel_size, self.strides, self.depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], self.kernel_size, self.strides, self.depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], self.kernel_size, 1, conv_type) for _ in range(self.depth)])

            # Output conv
            outputs = self.num_outputs if self.separate else self.num_outputs * len(self.instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(config.waveunet_length)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)
            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        '''
        x: [B, 1, T]
        '''
        
        # x = F.pad(
        #     x, 
        #     (self.shapes["output_start_frame"], 
        #     self.shapes["input_frames"] - self.config.waveunet_length - self.shapes["output_start_frame"]),
        #     "constant", 0
        # )
        x = F.pad(
            x, 
            (0,  
            self.shapes["input_frames"] - self.config.waveunet_length),
            "constant", 0
        )
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst : self.forward_module(x, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = torch.permute(out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs, :self.config.waveunet_length], (0,2,1))
            return out_dict

    def training_step(self, batch, batch_idx):
        self.train()
        self.device_type = next(self.parameters()).device
        if not self.check_flag:
            self.check_flag = True
        mixtures = np_to_pytorch(np.array(batch["mixture"])[:, None, :], self.device_type)
        sources = np_to_pytorch(np.array(batch["source"])[:, :, None], self.device_type)
        if len(mixtures) > 0:
            # train
            batch_output_dict = self(mixtures)["ALL"]
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
            mixtures = np_to_pytorch(split_mixtures[i:i + batch_size].copy()[:, None, :], self.device_type)
            sources = np_to_pytorch(split_sources[i:i + batch_size].copy()[:, :, None], self.device_type)

            if len(mixtures) > 0:
                # validate
                batch_output_dict = self(mixtures)["ALL"] # B T C
                preds =  batch_output_dict # B T C
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

        if (not self.init_opt) and (self.config.resume_checkpoint is not None):
            print("*******load opt and lr scheduler************")
            ckpt = torch.load(self.config.resume_checkpoint, map_location="cpu")


        optimizer = optim.Adam(
            self.parameters(), lr = self.config.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
        )

        if (not self.init_opt) and (self.config.resume_checkpoint is not None):
            optimizer.load_state_dict(ckpt["optimizer_states"][0])
        # scheduler=optim.lr_scheduler.CyclicLR(optimizer=optimizer,
        #     base_lr=1e-7, max_lr=self.config.learning_rate,
        #     step_size_up=2000, step_size_down=2000,
        #     cycle_momentum=False
        # )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max",
            factor=0.65, patience=3,verbose=True
        )

        if (not self.init_opt) and (self.config.resume_checkpoint is not None):
            scheduler.load_state_dict(ckpt["optimizer_states"][0])
        
        self.init_opt = True

        cop_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "mean_sdr"
            }
        } 
        return cop_dict