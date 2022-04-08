# Configuration of the model training
# Ke Chen
# 2022.02.05

exp_name = "bcbq_sop_specunet" # tasnet specunet
workspace = "/home/kechen/Research/KE_MCS/"
checkpointspace = "/projects/kechen/research/MCS/"
test_output = "wav_output"


dataset_path = "data/"
dataset_name = "BCBQ" # BCBQ # CantoriaDatase # AneStringDataset #ChoraleSingingDataset
split_file = "idx_string_ni.npy"
mix_name = "mix"
shift_tonality = False
sep_track = "soprano"
model_type = "MCS_SpecUNet" # "MCS_ConvTasNet" # "MCS_SpecUNet" # "MCS_DPIResUNet"

resume_checkpoint = "/projects/kechen/research/MCS/results/vocal_vor_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=17-mean_sdr=10.559-median_sdr=10.449.ckpt"
# "/projects/kechen/research/MCS/results/vocal_vor_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=17-mean_sdr=10.559-median_sdr=10.449.ckpt"
#"/projects/kechen/research/MCS/results/string_emb_alto_specunet/checkpoint/lightning_logs/version_2/checkpoints/l-epoch=33-mean_sdr=13.426-median_sdr=13.784.ckpt"



# performance string ni


# performance vor
# "/projects/kechen/research/MCS/results/performance_vocal_vor_tenor_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=90-mean_sdr=13.641-median_sdr=13.424.ckpt"
# "/projects/kechen/research/MCS/results/performance_vocal_vor_alto_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=76-mean_sdr=11.231-median_sdr=10.921.ckpt"
# "/projects/kechen/research/MCS/results/performance_vocal_vor_sop_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=96-mean_sdr=12.648-median_sdr=12.436.ckpt"
# "/projects/kechen/research/MCS/results/performance_vocal_vor_bass_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=80-mean_sdr=10.769-median_sdr=10.672.ckpt"

# string_ni
# "/projects/kechen/research/MCS/results/string_ni_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=93-mean_sdr=10.956-median_sdr=10.797.ckpt"
# "/projects/kechen/research/MCS/results/string_ni_tenor_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=95-mean_sdr=10.649-median_sdr=10.540.ckpt" 
# "/projects/kechen/research/MCS/results/string_ni_bass_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=73-mean_sdr=11.091-median_sdr=10.963.ckpt"
# "/projects/kechen/research/MCS/results/string_ni_alto_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=99-mean_sdr=11.412-median_sdr=11.113.ckpt"

# string emb
# "/projects/kechen/research/MCS/results/string_emb_sop_specunet/checkpoint/lightning_logs/version_10/checkpoints/l-epoch=95-mean_sdr=12.244-median_sdr=12.718.ckpt"
# "/projects/kechen/research/MCS/results/string_emb_tenor_specunet/checkpoint/lightning_logs/version_6/checkpoints/l-epoch=97-mean_sdr=12.818-median_sdr=13.184.ckpt"
# "/projects/kechen/research/MCS/results/string_emb_bass_specunet/checkpoint/lightning_logs/version_2/checkpoints/l-epoch=99-mean_sdr=13.412-median_sdr=13.708.ckpt" 
# "/projects/kechen/research/MCS/results/string_emb_alto_specunet/checkpoint/lightning_logs/version_2/checkpoints/l-epoch=33-mean_sdr=13.426-median_sdr=13.784.ckpt"

# vor 
# "/projects/kechen/research/MCS/results/vocal_vor_alto_specunet/checkpoint/lightning_logs/version_4/checkpoints/l-epoch=46-mean_sdr=10.497-median_sdr=10.190.ckpt"
# "/projects/kechen/research/MCS/results/vocal_vor_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=17-mean_sdr=10.559-median_sdr=10.449.ckpt"
# "/projects/kechen/research/MCS/results/vocal_vor_tenor_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=79-mean_sdr=12.369-median_sdr=12.250.ckpt" 
# "/projects/kechen/research/MCS/results/vocal_vor_bass_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=15-mean_sdr=9.635-median_sdr=9.538.ckpt" 

#"/projects/kechen/research/MCS/results/ctd_tenor_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=33-mean_sdr=-0.976-median_sdr=1.535.ckpt"

# "/projects/kechen/research/MCS/results/piano_noire_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=90-mean_sdr=9.283-median_sdr=9.732.ckpt"

# "/home/kechen/Research/KE_MCS/results/piano_noire_alto_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=7-mean_sdr=7.095-median_sdr=7.584.ckpt"

#"/home/kechen/Research/KE_MCS/results/string_emb_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=10-mean_sdr=0.547-median_sdr=0.358.ckpt"

# "/home/kechen/Research/KE_MCS/results/string_emb_tenor_specunet/checkpoint/lightning_logs/version_2/checkpoints/l-epoch=8-mean_sdr=1.701-median_sdr=1.298.ckpt" 

# "/home/kechen/Research/KE_MCS/results/string_emb_bass_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=31-mean_sdr=5.170-median_sdr=5.456.ckpt"

#"/home/kechen/Research/KE_MCS/results/vocal_vor_alto_specunet/checkpoint/lightning_logs/version_4/checkpoints/l-epoch=46-mean_sdr=10.497-median_sdr=10.190.ckpt"
# "/home/kechen/Research/KE_MCS/results/vocal_vor_tenor_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=79-mean_sdr=12.369-median_sdr=12.250.ckpt" 
# "/home/kechen/Research/KE_MCS/results/piano_noire_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=90-mean_sdr=9.283-median_sdr=9.732.ckpt" 
# "/home/kechen/Research/KE_MCS/results/piano_noire_tenor_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=92-mean_sdr=9.435-median_sdr=10.244.ckpt"
# "/home/kechen/Research/KE_MCS/results/vocal_vor_bass_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=15-mean_sdr=9.635-median_sdr=9.538.ckpt" 
# "/home/kechen/Research/KE_MCS/results/vocal_vor_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=17-mean_sdr=10.559-median_sdr=10.449.ckpt"
# "/home/kechen/Research/KE_MCS/results/piano_noire_bass_tasnet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=39-mean_sdr=10.285-median_sdr=11.097.ckpt"


loss_type = "mae"   # "si_snr" # "mae"

debug = False

batch_size = 8
learning_rate = 5e-4 # 1e-3 1e-4
max_epoch = 100
num_workers = 3
lr_scheduler_epoch = [20, 40, 60]
latent_dim = 2048 # deprecated
 
sample_rate = 22050 # emb is 44100 
clip_samples = sample_rate * 10 # audio_set 10-sec clip
segment_frames = 100
hop_samples = 441 # emb is 882
random_seed = 12412 # 444612 1536123 12412

# tasnet
tasnet_enc_dim = 512
tasnet_filter_length = 20
tasnet_win = 2
tasnet_feature_dim = 128
tasnet_hidden_dim = 512
tasnet_kernel = 3
tasnet_stack = 3
tasnet_layer = 8
tasnet_causal = False

# resunet
resunet_warmup_steps = 1000
resunet_reduce_lr_steps = 15000

# recon wave
recon_list = [
    "data/string_emb/h5_file/chorale_001_soprano.h5",
    "data/string_emb/h5_file/chorale_001.h5"
]

output_list = [
    # "data/recon/chorale_001_bass.wav",
    # "data/recon/chorale_001_alto.wav",
    # "data/recon/chorale_001_tenor.wav",
    "data/recon/chorale_001_soprano.wav",
    "data/recon/chorale_001.wav"
]