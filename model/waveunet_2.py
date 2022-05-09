import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import np_to_pytorch, evaluate_sdr
from losses import get_loss_func
import torch.distributed as dist
import torch.optim as optim
import numpy as np
from numpy.linalg import multi_dot, inv


class ResampleBlock(nn.Module):
    '''
    A Resample block groups a pair of downsampling and upsampling blocks together.
    The pair is for corresponding down and up blocks, the down block sends a tensor
    to the block below but also to its correspding up block.
    Down blocks do not correspond to those of Stoller et al.
    A down block starts at the decimation/downsampling operation and includes the subsequent convolution.
    This choice allows both the up and down blocks to share the same number of up and down channels
    which makes the interface of the ResampleBlock simpler.
    A ResampleBlock can also have the decimation/upsampling operations disabled, in this way
    we can use it to implement the two top level convolutions (also the output_channels in this case can be
    different if desired).
    Args:
        input_channels (int): the number of channels of the down block input.
        down_channels (int): the number of channels outgoing from the down block,
            which is also the number of channels going into the up block.
        down_kernel_size (int): filter size of down block convolution
        up_kernel_size (int): filter size of up block convolution
        inner_module (callable): module that consumes down block output and produces the input of the up block
        resample (boolean): whether to include the decimation and upsampling operations, default True
        output_channels (boolean): the number of channels outgoing from the up block, default None (same as input_channels)
    '''
    def __init__(self, input_channels, down_channels, down_kernel_size, up_kernel_size, inner_module,
            resample=True, output_channels=None, output_activation=None):
        super(ResampleBlock, self).__init__()
        self.resample = resample
        output_channels = input_channels if not output_channels else output_channels
        self.output_activation = output_activation if output_activation else _leaky_relu
        self.down_conv = nn.Conv1d(input_channels, down_channels, down_kernel_size)
        self.up_conv = nn.Conv1d(down_channels+input_channels, output_channels, up_kernel_size)
        torch.nn.init.xavier_uniform_(self.down_conv.weight)
        torch.nn.init.zeros_(self.down_conv.bias)
        torch.nn.init.xavier_uniform_(self.up_conv.weight)
        torch.nn.init.zeros_(self.up_conv.bias)
        self.inner_module = inner_module

    def forward(self, x):
        '''
        Applies a resampling transformation on x.
        Convolution operations usually require padding to transform borders,
        this module does not perform padding and instead produces an output smaller
        than the input.
        '''
        x, saved = self._down(x)
        x = self.inner_module(x)
        return self._up(x, saved)

    def _down(self, x):
        decimated = x[:, :, ::2] if self.resample else x
        return _leaky_relu(self.down_conv(decimated)), x

    def _up(self, x, saved):
        upsampled = F.interpolate(x, x.size()[2]*2-1, mode='linear') if self.resample else x
        enriched = torch.cat([centered_crop(saved, upsampled), upsampled], 1)
        return self.output_activation(self.up_conv(enriched))

def _leaky_relu(t):
    return F.leaky_relu(t, 0.3)

def centered_crop(t, target_shape):
    s, target_s = t.size()[2], target_shape.size()[2]
    d = s - target_s
    if d==0:
        return t
    return t[:,:,d//2:-d+d//2]

class MCS_Waveunet(pl.LightningModule):
    '''
    Creates a WaveUNet for source separation as described by Stoller et al.
    Args:
        input_channels (int): number of channels in input
        output_channels (int): number of channels in output
        down_kernel_size (int): kernel size used in down convolutions
        up_kernel_size (int): kernel size used in up convolutions
        depth (int): number of pairs of down and up blocks
        num_filters (int): number of additional convolution channels used at each deeper level
    '''
    def __init__(self, channels, config, dataset):
        super(MCS_Waveunet, self).__init__()


        input_channels = channels
        output_channels = channels
        down_kernel_size = 15
        up_kernel_size = config.waveunet_kernel
        depth = config.waveunet_blocks
        num_filters = 24
        self.dataset = dataset
        self.config = config
        self.check_flag = False
        
        # Create Resample blocks in a bottom to top direction
        block_stack = lambda x: x
        for i in range(depth):
            up_channels = (depth - i) * num_filters
            down_channels = (depth - i + 1) * num_filters
            block_stack = ResampleBlock(up_channels, down_channels, down_kernel_size, up_kernel_size, block_stack)

        self.top_block = ResampleBlock(input_channels, num_filters, down_kernel_size, 1, block_stack,
                resample=False, output_channels=output_channels, output_activation=torch.tanh)

    def forward(self, x):
        '''
        Applies a WaveUNet transformation to input tensor.
        Convolutions require context due to not performing padding when convolving borders (i.e. borders are not convolved :) ).
        Therefore the input is usually larger than the output, the difference depends on filter sizes and depth,
        see WaveUNetSizeCalculator to calculate the exact sizes.
        '''
        x = F.pad(
            x, 
            (0,  
            46130 - self.config.waveunet_length),
            "constant", 0
        )
        return self.top_block(x)
    
    def training_step(self, batch, batch_idx):
        self.train()
        self.device_type = next(self.parameters()).device
        if not self.check_flag:
            self.check_flag = True
        mixtures = np_to_pytorch(np.array(batch["mixture"])[:, None, :], self.device_type)
        sources = np_to_pytorch(np.array(batch["source"])[:, :, None], self.device_type)
        if len(mixtures) > 0:
            # train
            batch_output_dict = self(mixtures)
            print(batch_output_dict.size())
            exit()
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
                batch_output_dict = self(mixtures)["ALL"] # B C T
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

        # scheduler=optim.lr_scheduler.CyclicLR(optimizer=optimizer,
        #     base_lr=1e-7, max_lr=self.config.learning_rate,
        #     step_size_up=2000, step_size_down=2000,
        #     cycle_momentum=False
        # )

        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer, mode="max",
        #     factor=0.65, patience=3,verbose=True
        # )
        # cop_dict = {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "mean_sdr"
        #     }
        # } 

        return optimizer


class WaveUNetSizeCalculator:
    '''
    Calculates input and output size based on a requested output size.
    The calculated input size is larger than the output to include the context needed
    to convolve the borders.
    Plus the upsampling assumes that its inputs must be odd, so that constraint is also
    enforced.
    '''
    def __init__(self, downblock_kernel_size, upblock_kernel_size, depth):
        self.downblock_kernel_size = downblock_kernel_size
        self.upblock_kernel_size = upblock_kernel_size
        self.depth = depth
 
    def calculate_dimensions(self, requested_output):
        # Equations between inputs and outputs are encoded as linear transformations
        # See below
        downsample = [self._down_conv()] + [self._down()] * self.depth
        upsample = [self._up()] * self.depth + [np.eye(2)]

        # Input size needed before up blocks to produce requested output
        # Truncate in order to satisfy oddness constraint in upsampling
        input_pre_upblocks = _multi_dot(upsample).dot(np.array([requested_output, 1]))
        input_pre_upblocks = np.floor(input_pre_upblocks)

        output = inv(_multi_dot(upsample)).dot(input_pre_upblocks.T)
        input = _multi_dot(downsample).dot(input_pre_upblocks.T)

        _sanity_check(downsample + upsample, input, output)

        return (int(input[0]), int(output[0]))

    # Equations between input and output sizes are expressed as linear transformations
    # The linear transform receives a vector [output, 1] and produces [input, 1]
    # To be used with care as arithmetic should be carried in the integer domain
  
    def _down(self):
        return self._down_decimation().dot(self._down_conv())

    def _up(self):
        return self._up_upsample().dot(self._up_conv())

    def _down_conv(self):
        'input = output + downblock_kernel_size - 1'
        return np.array([[1, self.downblock_kernel_size-1], [0, 1]])

    def _down_decimation(self):
        'input = output * 2'
        return np.array([[2, 0], [0, 1]])

    def _up_upsample(self):
        '''
        input = output / 2 + 0.5, equivalent to
        input * 2 + 1 = output, on integer domain constrains output to be odd
        '''
        return np.array([[0.5, 0.5], [0, 1]])

    def _up_conv(self):
        'input = output + upblock_kernel_size - 1'
        return np.array([[1, self.upblock_kernel_size-1], [0, 1]])

def _sanity_check(ms, input, output):
    '''Check that equations hold in integer domain.'''
    t = output
    for m in reversed(ms):
        t = m.dot(t)
        assert np.array_equal(t, np.floor(t))
    assert np.array_equal(t, input)

def _multi_dot(ms):
    return ms[0] if len(ms)==1 else multi_dot(ms)