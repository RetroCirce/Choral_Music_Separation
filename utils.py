import os
from venv import create
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