import os
import config
from reaper_python import *


def render_action(dir = "", file = "test.wav", track = "mix"):
	RPR_GetSetProjectInfo_String(0, "RENDER_FILE", dir, True)
	RPR_GetSetProjectInfo_String(0, "RENDER_PATTERN", file , True)
	RPR_Main_OnCommand(40340, 0)
	if track == "tenor":
		command = 40940
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "bass":
		command = 40941
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "soprano":
		command = 40942
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "alto":
		command = 40943
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "girl":
		command = 40942
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
		command = 40943
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "boya":
		command = 40940
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
		command = 40941
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "hidis":
		command = 40941
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
		command = 40942
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "lowdis":
		command = 40940
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
		command = 40943
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "diff":
		command = 40940
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
		command = 40942
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	elif track == "zick":
		command = 40941
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
		command = 40943
		RPR_Main_OnCommand(command, 0)
		RPR_Main_OnCommand(40728, 0)
	RPR_Main_OnCommand(42230, 0)


def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)


def render_project(midi_file, tonality_index, output_path, output_name, override = True):
    output_file = os.path.join(output_path, output_name)
    track_name = config.global_config["track_name"]
    render_name = config.global_config["render_name"]
    for t in render_name:
        if os.path.exists(output_file + "_" + t + "_" + str(tonality_index) + ".wav"):
            if override:
                os.remove(output_file + "_" + t + "_" + str(tonality_index) + ".wav")
            else:
                RPR_ShowConsoleMsg("The file ", output_file + "_" + t + "_" + str(tonality_index) + ".wav"," already exist but the override option is False")
                return
    # erase previous midi tracks
    num_item = RPR_CountMediaItems(0)
    for i in range(num_item): 
        mi = RPR_GetMediaItem(0, 0)
        track = RPR_GetMediaItem_Track(mi)
        k = RPR_DeleteTrackMediaItem(track, mi)
    # insert new tracks
    for i in range(len(track_name)):
        file = midi_file + "_" +  track_name[i] + "_" + str(tonality_index) + ".mid"
        track = RPR_GetTrack(0, i + 1)
        RPR_SetOnlyTrackSelected(track)
        RPR_SetEditCurPos(0, True, False)
        RPR_InsertMedia(file, 0)
    for i in range(len(render_name)):
        solo_name = output_name + "_" + render_name[i] + "_" + str(tonality_index) + ".wav"
        render_action(output_path, solo_name, render_name[i])
    # render different tracks
    # render_action(output_path, output_name + "_" + str(tonality_index) + ".wav", -1)

render_path = config.render_path


project_name = RPR_GetSetProjectInfo_String(0, "PROJECT_NAME", "", False)[3]
project_name = project_name[:-4]
dataset_path = os.path.join(config.dataset_path, config.type_path, project_name)
render_path = os.path.join(config.render_path, config.type_path, project_name)

create_folder(render_path)

midi_files = list(set(d[:config.global_config["general_name_length"]] for d in os.listdir(dataset_path)))
midi_files.sort()

for midi_file in midi_files:
    for i in range(0, 12):
        path_tidx = os.path.join(dataset_path, midi_file + "_alto_" + str(i) + ".mid")
        if os.path.exists(path_tidx):
            render_project(os.path.join(dataset_path, midi_file), i, render_path, midi_file)
        else:
            break

    

