import os
import config
from reaper_python import *
from utils import render_action, create_folder


def render_project(midi_file, output_path, output_name, override = True):
    output_file = os.path.join(output_path, output_name)
    track_name = config.global_config["track_name"]
    if os.path.exists(output_file + ".wav"):
        if override:
            os.remove(output_file + ".wav")
        else:
            RPR_ShowConsoleMsg("The file ", output_name + ".wav"," already exist but the override option is False")
            return
    for t in track_name:
        if os.path.exists(output_file + "_" + t + ".wav"):
            if override:
                os.remove(output_file + "_" + t + ".wav")
            else:
                RPR_ShowConsoleMsg("The file ", output_name + "_" + t + ".wav"," already exist but the override option is False")
                return
    # erase previous midi tracks
    num_item = RPR_CountMediaItems(0)
    for i in range(num_item): 
        mi = RPR_GetMediaItem(0, 0)
        track = RPR_GetMediaItem_Track(mi)
        k = RPR_DeleteTrackMediaItem(track, mi)
    # insert new tracks
    for i in range(len(track_name)):
        file = midi_file + "_" +  track_name[i] + ".mid"
        track = RPR_GetTrack(0, i + 1)
        RPR_SetOnlyTrackSelected(track)
        RPR_SetEditCurPos(0, True, False)
        RPR_InsertMedia(file, 0)
    for i in range(len(track_name)):
        solo_name = output_name + "_" + track_name[i] + ".wav"
        render_action(output_path, solo_name, i)
    # render different tracks
    render_action(output_path, output_name, -1)

render_path = config.render_path


project_name = RPR_GetSetProjectInfo_String(0, "PROJECT_NAME", "", False)[3]
project_name = project_name[:-4]
dataset_path = os.path.join(config.dataset_path, project_name)
render_path = os.path.join(config.render_path, project_name)

create_folder(render_path)

midi_files = list(set(d[:config.global_config["general_name_length"]] for d in os.listdir(dataset_path)))
midi_files.sort()

for midi_file in midi_files:
    render_project(os.path.join(dataset_path, midi_file), render_path, midi_file)

