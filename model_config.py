# Configuration of the model training
# Ke Chen
# 2022.02.05

exp_name = "string_ni_bass_tasnet"
workspace = "/home/kechen/Research/KE_MCS/"

dataset_path = "data/"
dataset_name = "string_ni"
split_file = "idx_string_ni.npy"
sep_track = "bass"
model_type = "MCS_SpecUNet" # "MCS_ConvTasNet" # "MCS_SpecUNe"

resume_checkpoint = None


loss_type = "mae"   # "si_snr" # "mae"

debug = False

batch_size = 4 * 2
learning_rate = 1e-3 
max_epoch = 100
num_workers = 3
lr_scheduler_epoch = [20, 40, 60]
latent_dim = 2048 # deprecated
 
sample_rate = 22050
clip_samples = sample_rate * 10 # audio_set 10-sec clip
segment_frames = 200
hop_samples = 441
random_seed = 12412 # 444612 1536123 12412

# tasnet
tasnet_enc_dim = 512
tasnet_feature_dim = 128
tasnet_win = 2
tasnet_kernel = 3
tasnet_stack = 3
tasnet_layer = 8
tasnet_causal = False

