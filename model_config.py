# Configuration of the model training
# Ke Chen
# 2022.02.05

exp_name = "synthesize_urmp_bass_specunet" # tasnet specunet
workspace = "/home/la/kechen/Research/KE_MCS/"
checkpointspace = "/home/la/kechen/Research/KE_MCS/"
test_output = "wav_output"


dataset_path = "data/"
dataset_name = "urmp" # BCBQ # CantoriaDatase # AneStringDataset #ChoraleSingingDataset
split_file = "idx_string_ni.npy"
mix_name = "mix"
shift_tonality = True
sep_track = "bass"
model_type = "MCS_SpecUNet" 
# "MCS_ConvTasNet" # "MCS_SpecUNet" # "MCS_DPIResUNet" # "MCS_Waveunet"
 
resume_checkpoint = "/home/la/kechen/Research/KE_MCS/synthesized_midi_model/bass/l-epoch=62-mean_sdr=15.894-median_sdr=16.696.ckpt"
# "/home/la/kechen/Research/KE_MCS/results/string_emb_sop_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=70-mean_sdr=10.722-median_sdr=11.067.ckpt"

# performance_string_emb
# "/home/la/kechen/Research/KE_MCS/results/performance_string_emb_alto_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=79-mean_sdr=13.351-median_sdr=13.478.ckpt"
# "/home/la/kechen/Research/KE_MCS/results/performance_string_emb_bass_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=70-mean_sdr=11.887-median_sdr=12.540.ckpt"
# "/home/la/kechen/Research/KE_MCS/results/performance_string_emb_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=110-mean_sdr=11.562-median_sdr=12.289.ckpt"
# "/home/la/kechen/Research/KE_MCS/results/performance_string_emb_tenor_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=78-mean_sdr=11.695-median_sdr=12.046.ckpt"

# string emb
# "/home/la/kechen/Research/KE_MCS/results/string_emb_alto_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=55-mean_sdr=12.205-median_sdr=12.711.ckpt"
# "/home/la/kechen/Research/KE_MCS/results/string_emb_bass_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=43-mean_sdr=12.084-median_sdr=12.251.ckpt"
# "/home/la/kechen/Research/KE_MCS/results/string_emb_sop_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=70-mean_sdr=10.722-median_sdr=11.067.ckpt"
# "/home/la/kechen/Research/KE_MCS/results/string_emb_tenor_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=92-mean_sdr=11.523-median_sdr=12.118.ckpt"


# "/projects/kechen/research/MCS/results/vocal_vor_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=17-mean_sdr=10.559-median_sdr=10.449.ckpt"
#"/projects/kechen/research/MCS/results/string_emb_alto_specunet/checkpoint/lightning_logs/version_2/checkpoints/l-epoch=33-mean_sdr=13.426-median_sdr=13.784.ckpt"

# synthsize midi
# /projects/kechen/research/MCS/results/synthsize_alto_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=69-mean_sdr=18.732-median_sdr=21.137.ckpt
# /projects/kechen/research/MCS/results/synthsize_bass_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=62-mean_sdr=15.894-median_sdr=16.696.ckpt
# /projects/kechen/research/MCS/results/synthsize_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=89-mean_sdr=21.609-median_sdr=23.795.ckpt
# /projects/kechen/research/MCS/results/synthsize_tenor_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=91-mean_sdr=16.512-median_sdr=18.521.ckpt

# performance string ni


# performance vor
# "/projects/kechen/resear bch/MCS/results/performance_vocal_vor_tenor_specunet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=90-mean_sdr=13.641-median_sdr=13.424.ckpt"
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
# "/home/kechen/Research/KE_MCS/resultWs/vocal_vor_sop_specunet/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=17-mean_sdr=10.559-median_sdr=10.449.ckpt"
# "/home/kechen/Research/KE_MCS/results/piano_noire_bass_tasnet/checkpoint/lightning_logs/version_1/checkpoints/l-epoch=39-mean_sdr=10.285-median_sdr=11.097.ckpt"


loss_type = "mae"   # "si_snr" # "mae" # "mse"

debug = False

batch_size = 8
learning_rate = 1e-3 # 1e-3 1e-4
max_epoch = 100
num_workers = 0
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

# waveunet
waveunet_kernel = 5
waveunet_depth = 1
waveunet_stride = 4
waveunet_blocks = 6
waveunet_features = [32, 64, 128, 256, 512, 1024]
waveunet_res = "fixed"
waveunet_convtype = "gn"
waveunet_length = 22050 * 2

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