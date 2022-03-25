from re import A
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

class ChoraleSingingDataset(Dataset):
    def __init__(self, dataset_name, config, factor = 3, eval_mode = False):
        self.config = config
        self.factor = factor
        self.eval_mode = eval_mode
        self.dataset_path = os.path.join(config.dataset_path, dataset_name, "h5_file")
        
        if self.config.sep_track == "bass":
            self.keyword = "b"
        elif self.config.sep_track == "tenor":
            self.keyword = "t"
        elif self.config.sep_track == "soprano":
            self.keyword = "s"
        else:
            self.keyword = "a"
        # examing the file
        filelist = {}
        logging.info("examing the h5 file")
        for wav_file in tqdm(os.listdir(self.dataset_path)):
            name, voice = self.parse_data(wav_file)
            if voice.startswith(self.keyword) or voice.startswith("mix"):
                if name not in filelist:
                    filelist[name] = {}
                if voice.startswith(self.keyword):
                    filelist[name]["source"] = voice
                else:
                    filelist[name]["mixture"] = voice

        self.filelist = filelist
        self.idxs = list(self.filelist.keys())
        if self.eval_mode:
            factor = 1        
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))
    
    def parse_data(self, wav_file):
        words = wav_file.split("_")
        assert len(words) == 4, "the name format of the data is wrong!"
        name = words[0] + "_" + words[1] + "_" + words[3]
        voice = words[2]
        return name, voice

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
        h5_file_mixture = os.path.join(self.dataset_path, file_idx[:-5] + "_" + self.filelist[file_idx]["mixture"] + file_idx[-5:])
        h5_file_source = os.path.join(self.dataset_path, file_idx[:-5] + "_" + self.filelist[file_idx]["source"] + file_idx[-5:])
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


class Cantoria_Dataset(Dataset):
    def __init__(self, dataset_name, config, factor = 3, eval_mode = False):
        self.config = config
        self.factor = factor
        self.eval_mode = eval_mode
        self.dataset_path = os.path.join(config.dataset_path, dataset_name, "h5_file")
        
        if self.config.sep_track == "bass":
            self.keyword = "B"
        elif self.config.sep_track == "tenor":
            self.keyword = "T"
        elif self.config.sep_track == "soprano":
            self.keyword = "S"
        else:
            self.keyword = "A"
        # examing the file
        filelist = {}
        logging.info("examing the h5 file")
        for wav_file in tqdm(os.listdir(self.dataset_path)):
            name, voice = self.parse_data(wav_file)
            if voice.startswith(self.keyword) or voice == "Mix.h5":
                if name not in filelist:
                    filelist[name] = {}
                if voice.startswith(self.keyword):
                    filelist[name]["source"] = voice
                else:
                    filelist[name]["mixture"] = voice

        self.filelist = filelist
        self.idxs = list(self.filelist.keys())
        if self.eval_mode:
            factor = 1
            for k in self.idxs:
                if k.endswith("CEA"):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())
        else:
            for k in self.idxs:
                if not k.endswith("CEA"):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())   
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))
    
    def parse_data(self, wav_file):
        words = wav_file.split("_")
        assert len(words) == 3, "the name format of the data is wrong!"
        name = words[0] + "_" + words[1]
        voice = words[2]
        return name, voice

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
        h5_file_mixture = os.path.join(self.dataset_path, file_idx + "_" + self.filelist[file_idx]["mixture"])
        h5_file_source = os.path.join(self.dataset_path, file_idx + "_" + self.filelist[file_idx]["source"])
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


class DCS_Dataset(Dataset):
    def __init__(self, dataset_name, config, factor = 3, eval_mode = False):
        self.config = config
        self.factor = factor
        self.eval_mode = eval_mode
        self.dataset_path = os.path.join(config.dataset_path, dataset_name, "h5_file")
        
        if self.config.sep_track == "bass":
            self.keyword = "B"
        elif self.config.sep_track == "tenor":
            self.keyword = "T"
        elif self.config.sep_track == "soprano":
            self.keyword = "S"
        else:
            self.keyword = "A"
        # examing the file
        filelist = {}
        logging.info("examing the h5 file")
        for wav_file in tqdm(os.listdir(self.dataset_path)):
            name, voice = self.parse_data(wav_file)
            if voice.startswith(self.keyword) or voice.startswith("Mix"):
                if name not in filelist:
                    filelist[name] = {}
                if voice.startswith(self.keyword):
                    filelist[name]["source"] = voice
                else:
                    filelist[name]["mixture"] = voice

        self.filelist = filelist
        self.idxs = list(self.filelist.keys())
        if self.eval_mode:
            factor = 1        
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))
        print(self.filelist)
    
    def parse_data(self, wav_file):
        words = wav_file.split("_")
        assert len(words) == 6, "the name format of the data is wrong!"
        name = words[0] + "_" + words[1] + "_" + words[2] + "_" + words[3]
        voice = words[4] + "_" + words[5]
        return name, voice

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
        h5_file_mixture = os.path.join(self.dataset_path, file_idx + "_" + self.filelist[file_idx]["mixture"])
        h5_file_source = os.path.join(self.dataset_path, file_idx + "_" + self.filelist[file_idx]["source"])
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

