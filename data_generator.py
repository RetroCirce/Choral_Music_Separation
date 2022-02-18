import numpy as np
import torch
import logging
import os
import sys
import h5py
import csv
import time
import random
import json
from tqdm import tqdm
from utils import get_segment_bgn_end_frames
from datetime import datetime

from torch.utils.data import Dataset, Sampler

# polished LGSPDataset, the main dataset for procssing the audioset files
class AudioTrackDataset(Dataset):
    def __init__(self, idxs, config, factor = 3, eval_mode = False):
        self.config = config
        self.factor = factor
        self.eval_mode = eval_mode
        self.dataset_path = os.path.join(config.dataset_path, config.dataset_name)
        self.idxs = []
        # examing the file
        logging.info("examing the h5 file")
        for file_idx in tqdm(idxs):
            h5_file_mixture = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + ".h5")
            if os.path.exists(h5_file_mixture):
                self.idxs.append(file_idx)
        self.total_size = int(len(self.idxs) * factor)
        logging.info("total dataset size: %d" %(self.total_size))

        if self.eval_mode:
            logging.info("process evaluation set range")
            self.sample_range = {}
            for file_idx in tqdm(self.idxs):
                h5_file_mixture = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + ".h5")
                h5_file_source = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_" + config.sep_track + ".h5")
                with h5py.File(h5_file_mixture, "r") as hr:
                    mixture_length = len(hr["waveform"][:])
                with h5py.File(h5_file_source, "r") as hr:
                    source_length = len(hr["waveform"][:])
                bgn_f, end_f = get_segment_bgn_end_frames(min(mixture_length, source_length), config.hop_samples * config.segment_frames)
                self.sample_range[file_idx] = [bgn_f, end_f]
            logging.info("processed!")

    def generate_queue(self):
        if not self.eval_mode:
            random.shuffle(self.idxs)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "mixture": (clip_samples,),
            "source": (clip_samples,),
        }
        """
        file_idx = self.idxs[index % len(self.idxs)]
        h5_file_mixture = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + ".h5")
        h5_file_source = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_" + self.config.sep_track + ".h5")
        with h5py.File(h5_file_mixture, "r") as hr:
            audio_name = hr["audio_name"][()].decode()
            mixture = hr["waveform"][:]
        with h5py.File(h5_file_source, "r") as hr:
            source = hr["waveform"][:]
        if self.eval_mode:
            bgn_f = 0
            end_f = min(len(mixture), len(source)) # get the whole length
            # bgn_f, end_f = self.sample_range[file_idx]
        else:
            bgn_f, end_f = get_segment_bgn_end_frames(min(len(mixture), len(source)), self.config.hop_samples * self.config.segment_frames)
        mixture = mixture[bgn_f:end_f]
        source = source[bgn_f:end_f]
        data_dict = {}
        data_dict["audio_name"] = audio_name
        data_dict["mixture"] = mixture
        data_dict["source"] = source
        return data_dict

    def __len__(self):
        return self.total_size
