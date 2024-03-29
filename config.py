# Configuration of the data processing
# Ke Chen
# 2022.01.20


# midi channel control change 
cc_volume = 7

# data config
dataset_path = "D:/Research/audio source separation/Music Voice Source Separation/bach_chorale/" # must be absolute
type_path = "performance"
# "D:/Research/audio source separation/Music Voice Source Separation/bach_daw_demo/midi"
global_config = {
    "general_name_length": 11, # chorale_xxx
    "dynamic_tempo": False,
    "track_name": ["tenor", "bass", "soprano", "alto"],
    "render_name": ["tenor", "bass", "soprano", "alto"], # , "girl", "boya", "hidis", "lowdis", "diff", "zick"],
    # mix = s + a + t + b
    # girl = s + a
    # boya = t + b
    # hidis = s + b
    # lowdis = a + t
    # diff = s + t
    # zick = a + b
    # noalto = s + t + b...

    "octave_bias": [0,0,0,0],
    "phrase_indent": 60 / 90 * 0.95
}
local_config = [
    # {
    #     "name": "test_phrase",
    #     "note_range": [[47, 73],[33, 62],[59, 86],[52, 79]],
    #     "dynamic_vel": [60, 115],
    #     "legato": True,
    #     "word_control": [[1,2],[0,1,2],[0,1],[0,1,2,3]]
    # },
    # {
    #     "name": "piano_noire",
    #     "legato": False
    # },
    # {
    #     "name": "piano_grandeur"
    # },
    # {
    #     "name": "piano_maverick"
    # },
    # {
    #     "name": "string_ni", "octave_bias": [1,0,1,0], 
    #     "note_range": [[55, 101],[36, 84], [55, 101], [48, 91]],
    #     "dynamic_vel": [70, 120],
    #     "shift_tonality": 3,
    #     "word_control": [[36,38,40,41],[12,14,16,17],[36,38,40,41],[24,26,28,29]]
    # },
    # {
    #     "name": "string_emb", "octave_bias": [0,0,0,0],
    #     "note_range":[[48,91],[36,82],[55,101],[55,101]],
    #     "dynamic_vel": [80, 125],
    #     "shift_tonality": 3,
    #     "word_control": [[36, 38, 39, 40], [24, 26, 27, 28], [43, 45, 46, 47], [43, 45, 46, 47]]
    # },
    # {
    #     "name": "vocal_vor",
    #     "note_range": [[47, 73],[33, 62],[59, 86],[52, 79]],
    #     "shift_tonality": 3,
    #     "dynamic_vel": [80, 125],
    #     "legato": True,
    #     "word_control": [[1,2],[0,1,2],[0,1],[0,1,2,3]]
    # },
    {
        "name": "vocal_dominus",
        "note_range": [[43, 67], [40, 67],[55, 81], [55, 81]],
        "dynamic_vel": [100, 125],
        # "legato": True,
        "word_control": [[24,25,26,27,28,29,30],[24,25,26,27,28,29,30],[24,25,26,27,28,29,30],[24,25,26,27,28,29,30]]
    },
    # {
    #     "name": "vocal_saint",
    #     "note_range":[[46, 69],[38, 58],[58, 84],[53, 76]]
    # }
]

template_path = "D:/Research/audio source separation/Music Voice Source Separation/template/reaper" # must be absolute

# render config
render_path = "D:/Research/audio source separation/Music Voice Source Separation/output" # must be absolute
