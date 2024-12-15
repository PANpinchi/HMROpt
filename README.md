# HMROpt: 3D Human Pose and Shape Refinement via Hard Data Consistency
This repository contains the implementation for final project of the 	CSIE5428 Computer Vision Practice with Deep Learning course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this slides.


## Installation and Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
git clone --recursive https://github.com/statho/HMROpt.git
cd HMROpt
source install_environment.sh
```
Download the pretrained model weights, and annotations for the datasets by running the following:
```bash
source download_data.sh
```
This will download all necessary data files, and place them in `data/`. Alternatively, you can download them from [here](https://drive.google.com/file/d/1W53UMg8kee3HGRTNd2aNhMUew_kj36OH/view) and [here](https://drive.google.com/file/d/1f-D3xhQPMC9rwtaCVNoxtD4BQh4oQbY9/view). Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](https://smplify.is.tue.mpg.de/). Please go to the corresponding website and register to get access to the downloads section. Download the model, create a folder `data/smpl`, rename `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl`, and place it in `data/smpl/`.

Finally, if you wish to run the evaluation and/or training code, you will need to download the images/videos for the datasets. The instructions are mostly common with the description in [here](https://github.com/nkolot/ProHMR/blob/master/dataset_preprocessing/README.md). We provide the annotations for all datasets, so you will only need to download the images/videos. Edit the `IMG_DIR` in `hmropt/configs/datasets.yml` accordingly.


## Run demo on images
The following command will run ScoreHMR on top of HMR 2.0, using detected keypoints from ViTPose and bounding boxes from ViTDet, on all images in the specified `--img_folder`. For each image, it will save a rendering of all the reconstructed people together in the front view.
```bash
python demo_image.py \
    --img_folder "example_data/images" \
    --out_folder "demo_out/images"
```


## Run demo on videos
The following command will first run tracking with 4D-Humans and 2D keypoint detection with ViTPose, and then run temporal model fitting with ScoreHMR on the video specified with `--input_video`. It will create a video rendering of the reconstructed people in the folder specified by `--out_folder`. It will also save intermediate results from 4D-Humans and ViTPose.
```bash
python demo_video.py \
    --input_video "example_data/videos/breakdancing.mp4" \
    --out_folder "demo_out/videos"
```


## Evaluation
The evaluation code is contained in `eval/`. We provide evaluation on 3 different settings with the following scripts:
- `eval_keypoint_fitting.py` is used in single-frame model fitting evaluation.
- `eval_multiview.py` is used to evaluate the multi-view refinement.
- `eval_video.py` is used to evaluate ScoreHMR and ours HMROpt in temporal model fitting.

The evaluation code uses cached HMR 2.0 predictions, which can be downloaded from [here](https://drive.google.com/file/d/1m9lv9uDYosIVZ-u0R3GCy1J1wHYNVUMP/view) or by running:
```bash
source download_hmr2_preds.sh
```

We also provide example code for saving the HMR 2.0 and ProHMR predictions in the appropriate format in `data_preprocessing/cache_hmr2_preds.py`.

Evaluation code example:
```bash
python eval/eval_keypoint_fitting.py --dataset 3DPW-TEST --shuffle --use_default_ckpt
```
Running the above command will compute the MPJPE and Reconstruction Error before and after single-frame model fitting with ScoreHMR on the test set of 3DPW.


## Acknowledgements
The codes are based on [ScoreHMR](https://github.com/statho/ScoreHMR). Please also follow their licenses. Thanks for their awesome works.

## Environment
We implemented the code on an environment running Ubuntu 22.04.3, utilizing an Intel(R) Xeon(R) Gold 6342 CPU, along with a single NVIDIA RTX A6000 GPU equipped with 48 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{pan2024_hmropt,
    title  = {HMROpt: 3D Human Pose and Shape Refinement via Hard Data Consistency},
    author = {Pin-Chi Pan, Tzu Hsu, and Han-De Chen},
    url    = {https://github.com/PANpinchi/HMROpt},
    year   = {2024}
}
```
