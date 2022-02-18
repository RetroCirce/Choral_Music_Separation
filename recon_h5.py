import os
import numpy as np
import h5py
import model_config as config
import soundfile as sf

def recon_wav(filename, output):
    with h5py.File(filename, "r") as hr:
        wavefile = hr["waveform"][:]
        sf.write(output, wavefile, config.sample_rate)



for i in range(len(config.recon_list)):
    recon_wav(config.recon_list[i], config.output_list[i])

