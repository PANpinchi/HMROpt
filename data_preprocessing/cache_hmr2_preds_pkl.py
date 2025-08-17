import os
import torch
import argparse
import shutil
import pickle
import numpy as np
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils.renderer import cam_crop_to_full
from score_hmr.datasets import create_dataset
from score_hmr.utils import recursive_to
from score_hmr.configs import dataset_config
from tqdm import tqdm


DATASETS = ["H36M-MULTIVIEW"]
OUT_DIR = 'cache/hmr2b'
os.makedirs(OUT_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference.")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used for data loading.")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset_cfg = dataset_config()


# Load HMR 2.0b.
download_models(CACHE_DIR_4DHUMANS)
# Copy SMPL model to the appropriate path for HMR 2.0 if it does not exist.
if not os.path.isfile(f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl'):
    shutil.copy('data/smpl/SMPL_NEUTRAL.pkl', f'{CACHE_DIR_4DHUMANS}/data/smpl/')
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
model_cfg.defrost()
model_cfg.EXTRA.LOAD_PREDICTIONS = "hmr2"
model_cfg.freeze()
model = model.to(device)
model.eval()


for dset in DATASETS:
    print(f"Generating predictions on {dset}")
    pkl_path = f"{OUT_DIR}/{os.path.basename(dataset_cfg[dset]['DATASET_FILE'])}"
    dataset = create_dataset(model_cfg, dataset_cfg[dset], train=False)

    predictions_list = []
    for ii, batch in enumerate(tqdm(dataset)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_cam_t_full = cam_crop_to_full(
            out['pred_cam'],
            batch['box_center'].float(),
            batch['box_size'].float(),
            batch['img_size'].float(),
        ).cpu().numpy()

        temp_betas_ = out['pred_smpl_params']['betas'].cpu().numpy()
        temp_pose_ = torch.cat((out['pred_smpl_params']['global_orient'],
                                out['pred_smpl_params']['body_pose']), axis=1).cpu().numpy()
        temp_pred_cam_ = out['pred_cam'].cpu().numpy()
        temp_pred_cam_t_ = cam_crop_to_full(
                out['pred_cam'],
                batch['box_center'].float(),
                batch['box_size'].float(),
                batch['img_size'].float(),
            ).cpu().numpy()

        predictions = {
            'pred_betas': temp_betas_,
            'pred_pose': temp_pose_,
            'pred_cam': temp_pred_cam_,
            'pred_cam_t': temp_pred_cam_t_,
        }

        predictions_list.append(predictions)

    print('Saving pkl file...')
    # Save results to .pkl file
    with open(pkl_path, 'wb') as f:
        pickle.dump(predictions_list, f)

print(f"Results saved to {pkl_path}")

