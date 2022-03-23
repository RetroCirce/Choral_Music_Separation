import sys
import os

from pyparsing import nums
sys.path.append('..')
import numpy as np
import config
import pretty_midi as pyd
from tqdm import tqdm
import math
import random
import utils


def octave_shift(instrument, octave_bias):
    notes = instrument.notes[::]
    for i in range(len(notes)):
        notes[i].pitch += (12 * octave_bias)
    return notes

def maxmin_threshold(instrument, threshold):
    notes = instrument.notes[::]
    for i in range(len(notes)):
        if notes[i].pitch < threshold[0]:
            diff = math.ceil((threshold[0] - notes[i].pitch) / 12) * 12
            notes[i].pitch += diff
        elif notes[i].pitch > threshold[1]:
            diff = math.ceil((notes[i].pitch - threshold[1]) / 12) * 12
            notes[i].pitch -= diff
    return notes

def split_phrases(instrument, add_phrase = False):
    phrase_indent = config.global_config["phrase_indent"]
    notes = instrument.notes[::]
    notes.sort(key = lambda x:x.start)
    prev = notes[0].start
    phrase_sta = 0
    phrase_notes = [pyd.Note(100, 10, notes[0].start, notes[0].start + 0.25)]
    phrase_range = []
    for i in range(len(notes)):
        # detect phrase
        if notes[i].start - prev > phrase_indent:
            phrase_notes.append(
                pyd.Note(100, 10, notes[i].start, notes[i].start + 0.25)
            )
            phrase_range.append([phrase_sta, i - 1])
            phrase_sta = i
        prev = max(prev, notes[i].end)
    phrase_range.append([phrase_sta, len(notes) - 1])
    if add_phrase:
        notes += phrase_notes
    return notes, phrase_range

def add_legato(instrument, phrase_range, legato_factor = 0.2):
    phrase_indent = config.global_config["phrase_indent"]
    notes = instrument.notes[::]
    notes.sort(key = lambda x:x.start)
    for pr in phrase_range:
        for i in range(pr[0], pr[1] + 1):
            if i < pr[1]:
                if notes[i].pitch == notes[i+1].pitch:
                    continue
                if abs(notes[i].pitch - notes[i+1].pitch) < 5:
                    notes[i].end += (legato_factor * phrase_indent)
                else:
                    notes[i].end = notes[i+1].start - 0.01
            else:
                pass
                # if pr[1] + 1 != len(notes):
                #     notes[i].end += (phrase_indent * (1 - legato_factor))
    return notes

def add_dynamic_volume(instrument, dv, phrase_range):
    phrase_indent = config.global_config["phrase_indent"]
    notes = instrument.notes[::]
    notes.sort(key = lambda x:x.start)
    new_cc = instrument.control_changes[::]
    v0 = v1 = random.randint(dv[0], (dv[0] + dv[1]) // 2)
    for pr in phrase_range:
        sta = notes[pr[0]].start - phrase_indent * 0.02
        end = notes[pr[1]].end - phrase_indent * 0.02
        v0 = v1
        if v0 >= (dv[0] + dv[1]) // 2:
            v1 = random.randint(dv[0], (dv[0] + dv[1]) // 2)
        else:
            v1 = random.randint((dv[0] + dv[1]) // 2, dv[1])
        mode = random.randint(0,1)
        num_steps = int((end - sta) / 0.025)
        if mode == 0: # p -> f
            timesteps = np.linspace(sta, end, num_steps)
            velsteps = np.linspace(v0, v1, num_steps)
            
        if mode == 1: # p -> f -> p
            timesteps = np.linspace(sta, end, num_steps)
            c_step = random.randint(int(num_steps * 0.3), int(num_steps * 0.7))
            velsteps = np.concatenate(
                (
                    np.linspace(v0, v1, c_step),
                    np.linspace(v1, v0, num_steps - c_step)
                )
            ) 
            v1 = v0
        for j in range(len(timesteps)):
            new_cc.append(
                pyd.ControlChange(
                    config.cc_volume, int(round(velsteps[j])), timesteps[j]
                )
            )
    return new_cc

def add_word_control(instrument, phrase_range, words):
    phrase_indent = config.global_config["phrase_indent"]
    notes = instrument.notes[::]
    notes.sort(key = lambda x:x.start)
    new_notes = []
    for pr in phrase_range:
        sta = notes[pr[0]].start - phrase_indent * 0.2
        end = notes[pr[0]].start - phrase_indent * 0.1
        if sta < 0:
            sta = 0
        if end < 0:
            end = phrase_indent * 0.1
        word = random.choice(words)
        new_notes.append(
            pyd.Note(
                100, word, sta, end
            )
        )
    return new_notes

def delete_repeat(instrument, phrase_range):
    phrase_indent = config.global_config["phrase_indent"]
    notes = instrument.notes[::]
    notes.sort(key = lambda x:x.start)
    new_notes = []
    for pr in phrase_range:
        for i in range(pr[0], pr[1] + 1):
            if i < pr[1]:
                if notes[i].pitch == notes[i+1].pitch:
                    notes[i+1].start = notes[i].start
                    continue
            new_notes.append(notes[i])          
    return new_notes


dataset_path = os.path.join(config.dataset_path, "original")
track_name = config.global_config["track_name"]
midi_files = list(set(d[:config.global_config["general_name_length"]] for d in os.listdir(dataset_path)))
midi_files.sort()

for dataset in config.local_config:
    # ignore the piano track
    if "piano" in dataset["name"]:
        continue
    folder_name = os.path.join(config.dataset_path, dataset["name"])
    utils.create_folder(folder_name)
    midi_count = 0
    octave_bias = None
    note_range = None
    if "octave_bias" in dataset:
        octave_bias = dataset["octave_bias"]
    if "note_range" in dataset:
        note_range = dataset["note_range"]
    for midi_file in tqdm(midi_files):
        midi_queue = []
        for i in range(len(track_name)):
            midi = pyd.PrettyMIDI(os.path.join(dataset_path, midi_file + "_" + track_name[i] + ".mid"))
            # octave shift
            if octave_bias is not None:
                if octave_bias[i] != 0:
                    midi.instruments[0].notes = octave_shift(midi.instruments[0], octave_bias[i])
            # maxmin threshold
            if note_range is not None:
                midi.instruments[0].notes = maxmin_threshold(midi.instruments[0], note_range[i])
            # split phrase
            _, phrase_range = split_phrases((midi.instruments[0]))
            # add legato
            if "legato" in dataset and dataset["legato"] == True:
                midi.instruments[0].notes = add_legato(midi.instruments[0], phrase_range)
            # dynamic velocity
            if "dynamic_vel" in dataset:
                midi.instruments[0].control_changes = add_dynamic_volume(midi.instruments[0], dataset["dynamic_vel"], phrase_range)
            # word control
            if "word_control" in dataset:
                word_notes = add_word_control(midi.instruments[0],phrase_range,dataset["word_control"][i])
            # delete repeat
            if "legato" in dataset and dataset["legato"] == True:
                midi.instruments[0].notes = delete_repeat(midi.instruments[0], phrase_range)
            if "word_control" in dataset:
                midi.instruments[0].notes += word_notes
            midi_queue.append(midi)
        if len(midi_queue) == len(track_name):
            midi_count += 1
            for i in range(len(track_name)):
                midi_queue[i].write(os.path.join(folder_name, midi_file + "_" + track_name[i] + ".mid"))
    print("%d files in total, sucessfully processed %d file" %(len(midi_files), midi_count))




