# Automatic Generation of Music from midi and Fl-Studio template 
# Ke Chen
# 2022.01.20


import os
import time
import config
import numpy as np
import pyautogui as pya

dir_path = data_config.dataset_path


positions = {
    "search":[1225,422],
    "search_file":[628,525],
    "search_shortcut":[340,1557],
    "daw_file":[1651,22],
    "daw_export":[1704,228],
    "daw_wav":[1933,254],
    "daw_filename":[0,0],
    "daw_render":[1458,1051],
    "daw_alto":[504,408],
    "daw_alto_py":[592, 401],
    "daw_bass":[503,332],
    "daw_bass_py": [591,331],
    "daw_soprano":[509,365],
    "daw_soprano_py": [629,370],
    "daw_tenor":[505,293],
    "daw_tenor_py": [603,287],
    "daw_py_open": [327, 116],
    "daw_py_file": [388, 148],
    "daw_py_import": [632, 216],
    "daw_py_close": [2544, 116],
}

filename = list(set([d[:11] for d in os.listdir(dir_path)]))
filename.sort()

# print(pya.position())
# exit()
print("generation will begin at 5 seconds")
time.sleep(5.0)
pya.FAILSAFE = True
pya.PAUSE = 0.25

for f in filename:

    # # alto
    f_name = f + "_alto.mid"
    pya.click(x = positions["daw_alto"][0],y = positions["daw_alto"][1],button="right") 
    pya.click(x = positions["daw_alto_py"][0],y = positions["daw_alto_py"][1],button="left") 
    pya.click(x = positions["daw_py_open"][0],y = positions["daw_py_open"][1],button="left") 
    pya.click(x = positions["daw_py_file"][0],y = positions["daw_py_file"][1],button="left") 
    pya.click(x = positions["daw_py_import"][0],y = positions["daw_py_import"][1],button="left") 
    time.sleep(1.0)
    pya.click(x = positions["search"][0],y = positions["search"][1],button="left") 
    time.sleep(1.0)
    pya.write(f_name)
    time.sleep(1.0)
    pya.press('enter')
    time.sleep(1.0)
    pya.moveTo(positions["search_file"][0], positions["search_file"][1])
    pya.doubleClick()
    pya.click(x = positions["daw_py_close"][0],y = positions["daw_py_close"][1],button="left") 
    time.sleep(0.5)
    
    # # bass
    f_name = f + "_bass.mid"
    pya.click(x = positions["daw_bass"][0],y = positions["daw_bass"][1],button="right") 
    pya.click(x = positions["daw_bass_py"][0],y = positions["daw_bass_py"][1],button="left") 
    pya.click(x = positions["daw_py_open"][0],y = positions["daw_py_open"][1],button="left") 
    pya.click(x = positions["daw_py_file"][0],y = positions["daw_py_file"][1],button="left") 
    pya.click(x = positions["daw_py_import"][0],y = positions["daw_py_import"][1],button="left") 
    time.sleep(1.0)
    pya.click(x = positions["search"][0],y = positions["search"][1],button="left") 
    time.sleep(1.0)
    pya.write(f_name)
    time.sleep(1.0)
    pya.press('enter')
    time.sleep(1.0)
    pya.moveTo(positions["search_file"][0], positions["search_file"][1])
    pya.doubleClick()
    time.sleep(1.0)
    pya.click(x = positions["daw_py_close"][0],y = positions["daw_py_close"][1],button="left") 
    time.sleep(0.5)

    # # soprano
    f_name = f + "_soprano.mid"
    pya.click(x = positions["daw_soprano"][0],y = positions["daw_soprano"][1],button="right") 
    pya.click(x = positions["daw_soprano_py"][0],y = positions["daw_soprano_py"][1],button="left") 
    pya.click(x = positions["daw_py_open"][0],y = positions["daw_py_open"][1],button="left") 
    pya.click(x = positions["daw_py_file"][0],y = positions["daw_py_file"][1],button="left") 
    pya.click(x = positions["daw_py_import"][0],y = positions["daw_py_import"][1],button="left") 
    time.sleep(1.0)
    pya.click(x = positions["search"][0],y = positions["search"][1],button="left") 
    time.sleep(1.0)
    pya.write(f_name)
    time.sleep(1.0)
    pya.press('enter')
    time.sleep(1.0)
    pya.moveTo(positions["search_file"][0], positions["search_file"][1])
    pya.doubleClick()
    time.sleep(1.0)
    pya.click(x = positions["daw_py_close"][0],y = positions["daw_py_close"][1],button="left") 
    time.sleep(0.5)

    # tenor
    f_name = f + "_tenor.mid"
    pya.click(x = positions["daw_tenor"][0],y = positions["daw_tenor"][1],button="right") 
    pya.click(x = positions["daw_tenor_py"][0],y = positions["daw_tenor_py"][1],button="left") 
    pya.click(x = positions["daw_py_open"][0],y = positions["daw_py_open"][1],button="left") 
    pya.click(x = positions["daw_py_file"][0],y = positions["daw_py_file"][1],button="left") 
    pya.click(x = positions["daw_py_import"][0],y = positions["daw_py_import"][1],button="left") 
    time.sleep(1.0)
    pya.click(x = positions["search"][0],y = positions["search"][1],button="left") 
    time.sleep(1.0)
    pya.write(f_name)
    time.sleep(1.0)
    pya.press('enter')
    time.sleep(1.0)
    pya.moveTo(positions["search_file"][0], positions["search_file"][1])
    pya.doubleClick()
    time.sleep(1.0)
    pya.click(x = positions["daw_py_close"][0],y = positions["daw_py_close"][1],button="left") 
    time.sleep(0.5)
    
    # render 
    pya.click(x = positions["daw_file"][0],y = positions["daw_file"][1],button="left") 
    pya.click(x = positions["daw_export"][0],y = positions["daw_export"][1],button="left") 
    pya.click(x = positions["daw_wav"][0],y = positions["daw_wav"][1],button="left") 
    time.sleep(1.0)
    pya.write(f)
    time.sleep(1.0)
    pya.press('enter')
    pya.click(x = positions["daw_render"][0],y = positions["daw_render"][1],button="left") 
    
    time.sleep(10.0)
