# Choral Music Separation
 
## Introduction
This is the official implementation repository of "[Improving Choral Music Separation through Expressive Synthesized Data from Sampled Instruments](https://arxiv.org/abs/2209.02871)", ISMIR 2022.

In this project, we investigate the use of synthesized training data for the source separation task on real choral music. We make three contributions:
1. we provide an automated pipeline for synthesizing choral music data from sampled instrument plugins within controllable options for instrument expressiveness. This produces an 8.2-hour-long choral music dataset from the JSB Chorales Dataset and one can easily synthesize additional data.
2. we conduct an experiment to evaluate multiple separation models on available choral music separation datasets from previous work. To the best of our knowledge, this is the first experiment to comprehensively evaluate choral music separation.
3. Experiments demonstrate that the synthesized choral data is of sufficient quality to improve the model's performance on real choral music datasets. 

The datasets used in the paper are available at: [google drive](https://drive.google.com/drive/folders/1MoCWjofikml109Zb_wedFCDbYUHDqie_?usp=share_link)

The demo of the paper is shown at: [demo page](https://retrocirce.github.io/cms_demo/)

The released code is able to reproduce the results, but it is still organized in progress. The new folder structure with an instruction on the code will be uploaded soon.

## Citation
If you find this project and datasets useful, please cite our paper:
```
@article{cmske2022,
  title = {Improving Choral Music Separation through Expressive Synthesized Data from Sampled Instruments},
  author = {Ke Chen and Hao-Wen Dong and Yi Luo and Julian McAuley and Taylor Berg-Kirkpatrick and Miller Puckette and Shlomo Dubnov},
  booktitle = {Proceedings of the 23rd International Society for Music Information Retrieval Conference, {ISMIR}},
  year = {2022}
}
```
