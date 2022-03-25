import os
import json
import sre_compile
import librosa
import numpy as np
import random
import torch
import museval

from datetime import datetime
from reaper_python import *


def render_action(dir = "", file = "test.wav", track = -1):
	RPR_GetSetProjectInfo_String(0, "RENDER_FILE", dir, True)
	RPR_GetSetProjectInfo_String(0, "RENDER_PATTERN", file , True)
	RPR_Main_OnCommand(40340, 0)
	if track != -1:
		command = 40940 + track
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	RPR_Main_OnCommand(42230, 0)


def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

def collect_fn(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    return np_data_dict

def dump_config(config, filename, include_time = False):
    save_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_json = {}
    for key in dir(config):
        if not key.startswith("_"):
            config_json[key] = eval("config." + key)
    if include_time:
        filename = filename + "_" + save_time
    with open(filename + ".json", "w") as f:      
        json.dump(config_json, f ,indent=4)

def load_audio(wav_file, target_sr, is_mono = True):
    track, sr = librosa.load(wav_file, sr = None)
    if not is_mono:
        track = librosa.to_mono(track) 
    if sr != target_sr:
        track = librosa.resample(track, sr, target_sr)
    return track

def np_to_pytorch(x, device = None):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        x = torch.Tensor(x)
    return x.to(device)

def get_segment_bgn_end_frames(wav_length, segment_length):
    bgn_frame = random.randint(0, wav_length - segment_length - 1)
    end_frame = bgn_frame + segment_length
    return bgn_frame, end_frame

def calculate_sdr(ref, est, scaling=False):
    s = museval.evaluate(ref[None,:,None], est[None,:,None], win = len(ref), hop = len(ref))
    return s[0][0]

def calculate_silence_sdr(mixture, est):
    sdr = 10. * (
        np.log10(np.clip(np.mean(mixture ** 2), 1e-8, np.inf)) \
        - np.log10(np.clip(np.mean(est ** 2), 1e-8, np.inf)))
    return sdr

def evaluate_sdr(ref, est, class_ids, mix_type = "mixture"):
    sdr_results = []
    if mix_type == "silence":
        for i in range(len(ref)):
            sdr = calculate_silence_sdr(ref[i,:,0], est[i,:,0])
            sdr_results.append([sdr, class_ids[i]])
    else:
        for i in range(len(ref)):
            if np.sum(ref[i,:,0]) == 0 or np.sum(est[i,:,0]) == 0:
                continue
            else:
                sdr_c = calculate_sdr(ref[i,:,0], est[i,:,0], scaling = True)
            sdr_results.append([sdr_c, class_ids[i]])
    return sdr_results


def get_lr_lambda(step, warm_up_steps: int, reduce_lr_steps: int):
    r"""Get lr_lambda for LambdaLR. E.g.,
    .. code-block: python
        lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)
        from torch.optim.lr_scheduler import LambdaLR
        LambdaLR(optimizer, lr_lambda)
    Args:
        warm_up_steps: int, steps for warm up
        reduce_lr_steps: int, reduce learning rate by 0.9 every #reduce_lr_steps steps
    Returns:
        learning rate: float
    """
    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        return 0.9 ** (step // reduce_lr_steps)