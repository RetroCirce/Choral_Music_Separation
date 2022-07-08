from re import A
import numpy as np
from scipy.fftpack import shift
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

def build_mixture(mixture_name):
    if mixture_name == "" or mixture_name == "mix":
        return ["tenor","bass","soprano", "alto"]
    if mixture_name == "girl":
        return ["soprano", "alto"]
    if mixture_name == "boya":
        return ["tenor", "bass"]
    if mixture_name == "hidis":
        return ["soprano", "bass"]
    if mixture_name == "lowdis":
        return ["tenor", "alto"]
    if mixture_name == "zick":
        return ["alto","bass"]
    if mixture_name == "diff":
        return ["soprano", "tenor"]
    if mixture_name == "realrealmix":
        return ["mix"]
    return ["tenor","bass","soprano", "alto"]


class URMPDataset(Dataset):
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
                filelist[name]["source"] = voice
        self.filelist = filelist
        self.idxs = list(self.filelist.keys())
        if self.eval_mode:
            factor = 1
            for k in self.idxs:
                if self.is_train_data(k, 1):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())
        else:
            for k in self.idxs:
                if not self.is_train_data(k, 1):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())   
            
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))
    

    def is_train_data(self, k, degree = 1): # 10% 40% 70%
        if degree == 1:
            return k.startswith("1")
        elif degree == 2:
            return k.startswith("1") or k.startswith("3")
        else:
            return k.startswith("1") or k.startswith("2") or k.startswith("3")


    def parse_data(self, wav_file):
        words = wav_file.split("_")
        assert len(words) == 2, "the name format of the data is wrong!"
        name = words[0]
        voice = words[1]
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
        h5_file_source = os.path.join(self.dataset_path, file_idx + "_" + self.filelist[file_idx]["source"])
        
        with h5py.File(h5_file_source, "r") as hr:
            source = hr["waveform"][:]
            source_len = len(source)

        mixture_names = build_mixture(self.config.mix_name)
        audio_name = None
        mixture = None
        for mname in mixture_names:
            h5_file_mixture = os.path.join(self.dataset_path, file_idx + "_" + mname + ".h5")
            with h5py.File(h5_file_mixture, "r") as hr:
                audio_name = hr["audio_name"][()].decode() + "_" + self.config.mix_name
                temp_mixture = np.array(hr["waveform"][:])
                if len(temp_mixture) < source_len:
                    temp_mixture = np.pad(temp_mixture, (0, source_len - len(temp_mixture)), constant_values=(0))
                else:
                    temp_mixture = temp_mixture[:source_len]
                if mixture is None:
                    mixture = temp_mixture
                else:
                    mixture = mixture + temp_mixture

        mixture = list(mixture)
        if self.eval_mode:
            bgn_f = 0
            end_f = len(source) # get the whole length
            # bgn_f, end_f = self.sample_range[file_idx]
        else:
            bgn_f, end_f = get_segment_bgn_end_frames(len(source), self.config.hop_samples * self.config.segment_frames)
        mixture = mixture[bgn_f:end_f]
        source = source[bgn_f:end_f]
        data_dict = {}
        data_dict["audio_name"] = audio_name
        data_dict["mixture"] = mixture
        data_dict["source"] = source
        return data_dict

    def __len__(self):
        return self.total_size

class BCBQDataset(Dataset):
    def __init__(self, dataset_name, config, factor = 3, eval_mode = False):
        self.config = config
        self.factor = factor
        self.eval_mode = eval_mode
        self.dataset_path = os.path.join(config.dataset_path, dataset_name, "h5_file")
        
        if self.config.sep_track == "bass":
            self.keyword = "b_1ch"
        elif self.config.sep_track == "tenor":
            self.keyword = "t_1ch"
        elif self.config.sep_track == "soprano":
            self.keyword = "s_1ch"
        else:
            self.keyword = "a_1ch"
        # examing the file
        filelist = {}
        logging.info("examing the h5 file")
        for wav_file in tqdm(os.listdir(self.dataset_path)):
            name, voice = self.parse_data(wav_file)
            if voice.startswith(self.keyword):
                if name not in filelist:
                    filelist[name] = {}
                filelist[name]["source"] = voice
        self.filelist = filelist
        self.idxs = list(self.filelist.keys())
        if self.eval_mode:
            factor = 1
            for k in self.idxs:
                if self.is_train_data(k, 3): 
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())
        else:
            for k in self.idxs:
                if not self.is_train_data(k, 3):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())       
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))

    def is_train_data(self, k, degree = 1): # 10% 40% 70%
        if degree == 1:
            return k.startswith("1_") or k.startswith("13_") or k.startswith("18_")
        elif degree == 2:
            return k.startswith("1_") or k.startswith("13_") or k.startswith("18_") or \
                k.startswith("4_") or k.startswith("9_") or k.startswith("11_") or \
                k.startswith("5_") or k.startswith("19_") or k.startswith("22_") or k.startswith("6_")
        else:
            return k.startswith("1_") or k.startswith("13_") or k.startswith("18_") or \
                k.startswith("4_") or k.startswith("9_") or k.startswith("11_") or \
                k.startswith("5_") or k.startswith("19_") or k.startswith("22_") or k.startswith("6_") or \
                k.startswith("2_") or k.startswith("14_") or k.startswith("15_") or k.startswith("20_") or \
                k.startswith("8_") or k.startswith("21_") or k.startswith("25_") or k.startswith("26_")



    def parse_data(self, wav_file):
        words = wav_file.split("_")
        assert len(words) == 5, "the name format of the data is wrong!"
        name = words[0] + "_" + words[1] + "_" + words[2] 
        voice = words[3] + "_" + words[4]
        return name, voice

    def generate_queue(self):
        if not self.eval_mode:
            random.shuffle(self.idxs)
    
    def BCBQ_mixture(self, mixture_name):
        if mixture_name == "" or mixture_name == "mix":
            return ["t","b","s", "a"]
        if mixture_name == "girl":
            return ["s", "a"]
        if mixture_name == "boya":
            return ["t", "b"]
        if mixture_name == "hidis":
            return ["s", "b"]
        if mixture_name == "lowdis":
            return ["t", "a"]
        if mixture_name == "zick":
            return ["a","b"]
        if mixture_name == "diff":
            return ["s", "t"]
        if mixture_name == "realrealmix":
            return ["mix"]
        return ["t","b","s","a"]       

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
        h5_file_source = os.path.join(self.dataset_path, file_idx + "_" + self.filelist[file_idx]["source"])
        with h5py.File(h5_file_source, "r") as hr:
            source = hr["waveform"][:]
            source_len = len(source)

        mixture_names = self.BCBQ_mixture(self.config.mix_name)
        audio_name = None
        mixture = None
        for mname in mixture_names:
            h5_file_mixture = os.path.join(self.dataset_path, file_idx + "_" + mname + "_1ch.h5")
            with h5py.File(h5_file_mixture, "r") as hr:
                audio_name = hr["audio_name"][()].decode() + "_" + self.config.mix_name
                if mixture is None:
                    mixture = np.array(hr["waveform"][:source_len])
                else:
                    mixture = mixture + np.array(hr["waveform"][:source_len])

        mixture = list(mixture)
        if self.eval_mode:
            bgn_f = 0
            end_f = len(source) # get the whole length
            # bgn_f, end_f = self.sample_range[file_idx]
        else:
            bgn_f, end_f = get_segment_bgn_end_frames(len(source), self.config.hop_samples * self.config.segment_frames)
        mixture = mixture[bgn_f:end_f]
        source = source[bgn_f:end_f]
        data_dict = {}
        data_dict["audio_name"] = audio_name
        data_dict["mixture"] = mixture
        data_dict["source"] = source
        return data_dict

    def __len__(self):
        return self.total_size



class AneStringDataset(Dataset):
    def __init__(self, dataset_name, config, factor = 3, eval_mode = False):
        self.config = config
        self.factor = factor
        self.eval_mode = eval_mode
        self.dataset_path = os.path.join(config.dataset_path, dataset_name, "h5_file")
        
        if self.config.sep_track == "bass":
            self.keyword = "bass"
        elif self.config.sep_track == "tenor":
            self.keyword = "tenor"
        elif self.config.sep_track == "soprano":
            self.keyword = "soprano"
        else:
            self.keyword = "alto"
        # examing the file
        filelist = {}
        logging.info("examing the h5 file")
        for wav_file in tqdm(os.listdir(self.dataset_path)):
            name, voice = self.parse_data(wav_file)
            if voice.startswith(self.keyword):
                if name not in filelist:
                    filelist[name] = {}
                filelist[name]["source"] = voice
        self.filelist = filelist
        self.idxs = list(self.filelist.keys())
        if self.eval_mode:
            factor = 1
            for k in self.idxs:
                if k.endswith("001"):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())
        else:
            for k in self.idxs:
                if not k.endswith("001"):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())       
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))
    
    def parse_data(self, wav_file):
        words = wav_file.split("_")
        assert len(words) == 2, "the name format of the data is wrong!"
        name = words[0]
        voice = words[1]
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
        h5_file_source = os.path.join(self.dataset_path, file_idx + "_" + self.filelist[file_idx]["source"])
        with h5py.File(h5_file_source, "r") as hr:
            source = hr["waveform"][:]
            source_len = len(source)

        mixture_names = build_mixture(self.config.mix_name)
        audio_name = None
        mixture = None
        for mname in mixture_names:
            h5_file_mixture = os.path.join(self.dataset_path, file_idx + "_" + mname + ".h5")
            with h5py.File(h5_file_mixture, "r") as hr:
                audio_name = hr["audio_name"][()].decode() + "_" + self.config.mix_name
                if mixture is None:
                    mixture = np.array(hr["waveform"][:source_len])
                else:
                    mixture = mixture + np.array(hr["waveform"][:source_len])

        mixture = list(mixture)
        if self.eval_mode:
            bgn_f = 0
            end_f = len(source) # get the whole length
            # bgn_f, end_f = self.sample_range[file_idx]
        else:
            bgn_f, end_f = get_segment_bgn_end_frames(len(source), self.config.hop_samples * self.config.segment_frames)
        mixture = mixture[bgn_f:end_f]
        source = source[bgn_f:end_f]
        data_dict = {}
        data_dict["audio_name"] = audio_name
        data_dict["mixture"] = mixture
        data_dict["source"] = source
        return data_dict

    def __len__(self):
        return self.total_size



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
            for k in self.idxs:
                if self.is_train_data(k, 3):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())
        else:
            for k in self.idxs:
                if not self.is_train_data(k, 3):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())   
            
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))
    

    def is_train_data(self, k, degree = 1): # 10% 40% 70%
        if degree == 1:
            return k.startswith("CSD_ER")
        elif degree == 2:
            return k.startswith("CSD_ER") or k.startswith("CSD_LI_1") or k.startswith("CSD_LI_2") 
        else:
            return k.startswith("CSD_ER") or k.startswith("CSD_LI_1") or k.startswith("CSD_LI_2") or k.startswith("CSD_ND_3") or k.startswith("CSD_ND_4") 


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
                if self.is_train_data(k, 3):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())
        else:
            for k in self.idxs:
                if not self.is_train_data(k, 3):
                    self.filelist.pop(k)
            self.idxs = list(self.filelist.keys())   
        self.total_size = int(len(self.filelist) * factor)
        logging.info("total dataset size: %d" %(self.total_size))

    def is_train_data(self, k, degree = 1): # 10% 40% 70%
        if degree == 1:
            return k.endswith("CEA")
        elif degree == 2:
            return k.endswith("CEA") or k.endswith("HCB") or k.endswith("LBM1") or k.endswith("LJT2") or k.endswith("YSM") or k.endswith("THM")
        else:
            return k.endswith("CEA") or k.endswith("HCB") or k.endswith("LBM1") or k.endswith("LJT2") or k.endswith("YSM") or k.endswith("THM") or k.endswith("SSS") or k.endswith("LNG")

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
        self.idxs_tonality = {}
        # examing the file
        logging.info("examing the h5 file")
        for file_idx in tqdm(idxs):
            if self.config.shift_tonality:
                h5_file_test = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_alto_0.h5")
            else:
                h5_file_test = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_alto.h5")
            if os.path.exists(h5_file_test):
                self.idxs.append(file_idx) 
                if self.config.shift_tonality:
                    self.idxs_tonality[file_idx] = [0]
                    for j in range(1,12):
                        h5_file_test = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_alto_" + str(j) + ".h5")
                        if os.path.exists(h5_file_test):
                            self.idxs_tonality[file_idx].append(j)  
        if self.config.shift_tonality:
            print(self.idxs_tonality)             
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
        if self.config.shift_tonality:
            if self.eval_mode:
                tonality = 0    
            else:
                tonality = random.choice(self.idxs_tonality[file_idx])
            h5_file_source = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_" + self.config.sep_track + "_" + str(tonality) + ".h5")
        else:
            h5_file_source = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_" + self.config.sep_track + ".h5")
        
        with h5py.File(h5_file_source, "r") as hr:
            source = hr["waveform"][:]
            source_len = len(source)
        mixture_names = build_mixture(self.config.mix_name)
        audio_name = None
        mixture = None

        for mname in mixture_names:
            if self.config.shift_tonality:
                h5_file_mixture = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_" + mname + "_" + str(tonality) + ".h5")
            else:
                h5_file_mixture = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + "_" + mname + ".h5")
            if mname == "mix" and (not os.path.exists(h5_file_mixture)):
                h5_file_mixture = os.path.join(self.dataset_path, "h5_file", "chorale_" + file_idx + ".h5")
            with h5py.File(h5_file_mixture, "r") as hr:
                audio_name = hr["audio_name"][()].decode() + "_" + self.config.mix_name
                if mixture is None:
                    mixture = np.array(hr["waveform"][:source_len])
                else:
                    mixture = mixture + np.array(hr["waveform"][:source_len])
        mixture = list(mixture)
        if self.eval_mode:
            bgn_f = 0
            end_f = len(mixture) # get the whole length
            # bgn_f, end_f = self.sample_range[file_idx]
        else:
            bgn_f, end_f = get_segment_bgn_end_frames(len(mixture), self.config.hop_samples * self.config.segment_frames)
        mixture = mixture[bgn_f:end_f]
        source = source[bgn_f:end_f]
        data_dict = {}
        data_dict["audio_name"] = audio_name
        data_dict["mixture"] = mixture
        data_dict["source"] = source
        return data_dict

    def __len__(self):
        return self.total_size

