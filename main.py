# Ke Chen
# The Main Script

import os

from py import process
# this is to avoid the sdr calculation from occupying all cpus
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import h5py
import numpy as np
import argparse
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import model_config as config

from utils import collect_fn, dump_config, create_folder, load_audio
from data_generator import AudioTrackDataset
from model.specunet import MCS_SpecUNet
from model.convtasnet import MCS_ConvTasNet

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


import warnings
warnings.filterwarnings("ignore")



class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num, config):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num
        self.config = config

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle = False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = train_sampler,
            collate_fn = collect_fn
        )
        return train_loader
    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = 1, # config.batch_size // self.device_num, for the sdr calculation on one song
            shuffle = False,
            sampler = eval_sampler,
            collate_fn = collect_fn
        )
        return eval_loader
    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = 1, # config.batch_size // self.device_num, for the sdr calculation on one song
            shuffle = False,
            sampler = test_sampler,
            collate_fn = collect_fn
        )
        return test_loader

# weight average will perform in the given folder
# It will output one model checkpoint, which avergas the weight of all models in the folder
def weight_average():
    model_ckpt = []
    model_files = os.listdir(config.wa_model_folder)
    wa_ckpt = {
        "state_dict": {}
    }

    for model_file in model_files:
        model_file = os.path.join(config.esm_model_folder, model_file)
        model_ckpt.append(torch.load(model_file, map_location="cpu")["state_dict"])
    keys = model_ckpt[0].keys()
    for key in keys:
        model_ckpt_key = torch.cat([d[key].float().unsqueeze(0) for d in model_ckpt])
        model_ckpt_key = torch.mean(model_ckpt_key, dim = 0)
        assert model_ckpt_key.shape == model_ckpt[0][key].shape, "the shape is unmatched " + model_ckpt_key.shape + " " + model_ckpt[0][key].shape
        wa_ckpt["state_dict"][key] = model_ckpt_key
    torch.save(wa_ckpt, config.wa_model_path)

def process_audio(process_main_track = False):
    dataset_path = os.path.join(config.dataset_path, config.dataset_name)
    idxs = np.load(os.path.join(config.dataset_path, config.split_file), allow_pickle = True)
    idxs = idxs.item()
    count = [0,0,0]

    create_folder(os.path.join(dataset_path, "h5_file"))
    for i, key in enumerate(["train","validate","test"]):
        for file_idx in tqdm(idxs[key]):
            if process_main_track:
                wav_file = os.path.join(dataset_path, "chorale_" + file_idx + ".wav")
                h5_file = os.path.join(dataset_path, "h5_file", "chorale_" + file_idx + ".h5")
            else:
                wav_file = os.path.join(dataset_path, "chorale_" + file_idx + "_" + config.sep_track + ".wav")
                h5_file = os.path.join(dataset_path, "h5_file", "chorale_" + file_idx + "_" + config.sep_track + ".h5")
            if os.path.exists(wav_file):
                track = load_audio(wav_file = wav_file, target_sr = config.sample_rate)
                count[i] += 1
                with h5py.File(h5_file, "w") as hw:
                    hw.create_dataset("audio_name", data = file_idx, dtype="S20")
                    hw.create_dataset("waveform", data = track)
    print("train | validate | test:", count)

# test the separation model, mainly in musdb
def test():
    exit()
    # set exp settings
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda")
    assert config.test_key is not None, "there should be a separate key"
    create_folder(config.wave_output_path)
    # use musdb as testset
    test_data = np.load(config.testset_path, allow_pickle = True)
    print(len(test_data))
    mus_tracks = []
    # in musdb, all fs is the same (44100)
    # load the dataset
    for track in test_data:
        temp = []
        mixture = track["mixture"]
        temp.append(mixture)
        for dickey in config.test_key:
            source = track[dickey]
            temp.append(source)
        temp = np.array(temp)
        print(temp.shape)
        mus_tracks.append(temp)
    print(len(mus_tracks))
    dataset = MusdbDataset(tracks = mus_tracks)
    loader = DataLoader(
        dataset = dataset,
        num_workers = 1,
        batch_size = 1,
        shuffle = False
    )
    assert config.resume_checkpoint is not None, "there should be a saved model when inferring"
    
    sed_model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        in_chans=1,
        num_classes=htsat_config.classes_num,
        window_size=htsat_config.htsat_window_size,
        config = htsat_config,
        depths = htsat_config.htsat_depth,
        embed_dim = htsat_config.htsat_dim,
        patch_stride=htsat_config.htsat_stride,
        num_heads=htsat_config.htsat_num_head
    )
    at_model = SEDWrapper(
        sed_model = sed_model, 
        config = htsat_config,
        dataset = None
    )
    ckpt = torch.load(htsat_config.resume_checkpoint, map_location="cpu")
    at_model.load_state_dict(ckpt["state_dict"])
    trainer = pl.Trainer(
        gpus = 1
    )
    avg_at = None
    # obtain the query of four stems from the training set
    if config.infer_type == "mean":
        avg_data = np.load(config.testavg_path, allow_pickle = True)[:90]
        print(len(avg_data))
        avgmus_tracks = []
        # in musdb, all fs is the same (44100)
        # load the dataset
        for track in avg_data:
            temp = []
            mixture = track["mixture"]
            temp.append(mixture)
            for dickey in config.test_key:
                source = track[dickey]
                temp.append(source)
            temp = np.array(temp)
            print(temp.shape)
            avgmus_tracks.append(temp)
        print(len(avgmus_tracks))
        avg_dataset = MusdbDataset(tracks = avgmus_tracks)
        avg_loader = DataLoader(
            dataset = avg_dataset,
            num_workers = 1,
            batch_size = 1,
            shuffle = False
        )
        at_wrapper = AutoTaggingWarpper(
            at_model = at_model,
            config = config,
            target_keys = config.test_key
        )
        trainer.test(at_wrapper, test_dataloaders = avg_loader)
        avg_at = at_wrapper.avg_at
    
    model = ZeroShotASP(
        channels = 1, config = config, 
        at_model = at_model, 
        dataset = dataset
    )
    ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict= False)
    exp_model = SeparatorModel(
        model = model,
        config = config,
        target_keys = config.test_key,
        avg_at = avg_at,
        using_wiener = config.using_wiener
    )
    trainer.test(exp_model, test_dataloaders = loader)

def train():
    # set exp settings
    # device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda")

    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)

    
    idxs = np.load(os.path.join(config.dataset_path, config.split_file), allow_pickle = True)
    idxs = idxs.item()
    train_idxs = idxs["train"]
    validate_idxs = idxs["train"][:30]
    # validate_idxs = idxs["validate"]
    test_idxs = idxs["test"]

    # set exp folder
    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
    
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)
        
    # load data
    # import dataset LGSPDataset (latent general source separation) and sampler
    dataset = AudioTrackDataset(
        idxs=train_idxs,
        config=config,
        factor=20,
        eval_mode=False
    )
    eval_dataset = AudioTrackDataset(
        idxs=validate_idxs,
        config=config,
        factor=1,
        eval_mode=True
    )

    audioset_data = data_prep(train_dataset=dataset,eval_dataset=eval_dataset,device_num=device_num, config=config)
    checkpoint_callback = ModelCheckpoint(
        monitor = "median_sdr",
        filename='l-{epoch:d}-{mean_sdr:.3f}-{median_sdr:.3f}',
        save_top_k = 3,
        mode = "max"
    )
   
    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir = checkpoint_dir,
        gpus = device_num,
        # val_check_interval = 1,
        check_val_every_n_epoch = 1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        accelerator = "ddp" if device_num > 1 else None,
        resume_from_checkpoint = None, #config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=5.0,
        num_sanity_val_steps = 0
    )
    model_type = eval(config.model_type)
    model = model_type(
        channels=1,
        config=config,
        dataset=dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    # trainer.test(model, datamodule = audioset_data)
    trainer.fit(model, audioset_data)

def main():
    parser = argparse.ArgumentParser(description="latent genreal source separation parser")
    subparsers = parser.add_subparsers(dest = "mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    parser_process_audio = subparsers.add_parser("process_audio")
    args = parser.parse_args()
    # default settings
    logging.basicConfig(level=logging.INFO) 
    pl.utilities.seed.seed_everything(seed = config.random_seed)

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "process_audio":
        process_audio()
    else:
        raise Exception("Error Mode!")
    

if __name__ == '__main__':
    main()