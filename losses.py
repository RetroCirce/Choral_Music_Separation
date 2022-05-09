import torch.nn.functional as F
import torch
import numpy as np


def mae(input, target):
    return torch.mean(torch.abs(input - target))


def logmae_wav(model, output_dict, target):
    loss = torch.log10(torch.clamp(mae(output_dict['wav'], target), 1e-8, np.inf))
    return loss


def max_si_snr(input, target, eps = 1e-8):
    assert input.size() == target.size()
    B, C, T = target.size()
    # print(B, C, T)

    # zero-mean norm
    mean_target = torch.sum(target, dim = 2, keepdim=True) / T
    # print(mean_target.size())
    mean_input = torch.sum(input, dim = 2, keepdim=True) / T
    # print(mean_input.size())
    zero_mean_target = target - mean_target
    zero_mean_input = input - mean_input
    # print(zero_mean_target.size())
    # print(zero_mean_input.size())

    # Si-SNR
    s_target = torch.unsqueeze(zero_mean_target, dim = 1)
    # print(s_target.size())
    s_input = torch.unsqueeze(zero_mean_input, dim = 2)
    # print(s_input.size())
    pair_wise_dot = torch.sum(s_input * s_target, dim = 3, keepdim=True)
    # print(pair_wise_dot.size())
    s_target_energy = torch.sum(s_target ** 2, dim = 3, keepdim=True)
    # print(s_target_energy.size())
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy
    # print(pair_wise_proj.size())
    e_noise = s_input - pair_wise_proj
    # print(e_noise.size())
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim = 3) / (torch.sum(e_noise ** 2, dim = 3) + eps) 
    
 
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)
    # print(pair_wise_si_snr.size())

    k = pair_wise_si_snr.squeeze(1).squeeze(1)
    # print(k.size())
    loss = 0 - torch.mean(k) 

    return loss 

def get_loss_func(loss_type):
    if loss_type == 'logmae_wav':
        return logmae_wav
    elif loss_type == 'mae':
        return mae
    elif loss_type == 'si_snr':
        return max_si_snr
    elif loss_type == 'mse':
        return torch.nn.MSELoss()
    else:
        raise Exception('Incorrect loss_type!')