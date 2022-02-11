# Configuration of the model training
# Ke Chen
# 2022.02.05

exp_name = "string_ni_sop"
workspace = "/home/kechen/Research/KE_MCS/"

dataset_path = "data/"
dataset_name = "string_ni"
split_file = "idx_string_ni.npy"
sep_track = "soprano"

resume_checkpoint = None


loss_type = "mae"

debug = False

batch_size = 12 * 2
learning_rate = 1e-3 # 3e-4 is also workable
max_epoch = 100
num_workers = 3
lr_scheduler_epoch = [90, 110]
latent_dim = 2048 # deprecated
 
sample_rate = 22050
clip_samples = sample_rate * 10 # audio_set 10-sec clip
segment_frames = 100
hop_samples = 441
random_seed = 12412 # 444612 1536123 12412

