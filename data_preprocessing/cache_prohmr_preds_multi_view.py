import os
import torch
import argparse
import shutil
import pickle
import numpy as np
from hmr2.configs import CACHE_DIR_4DHUMANS
from prohmr.models import ProHMR
from prohmr.configs import prohmr_config
from hmr2.utils.renderer import cam_crop_to_full
from score_hmr.datasets import create_dataset
from score_hmr.utils import recursive_to
from score_hmr.configs import dataset_config
from tqdm import tqdm


DATASETS = ["H36M-MULTIVIEW"]
OUT_DIR = 'cache/prohmr'
os.makedirs(OUT_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset_cfg = dataset_config()

# Load proHMR.
print("Loading proHMR model...")
model_cfg = prohmr_config()
model_cfg.defrost()
model_cfg.EXTRA.LOAD_PREDICTIONS = "hmr2"
model_cfg.freeze()

prohmr_model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg)
prohmr_model = prohmr_model.to(device)
prohmr_model.eval()

for dset in DATASETS:
    print(f"Generating predictions on {dset}")
    pkl_path = f"{OUT_DIR}/{os.path.basename(dataset_cfg[dset]['DATASET_FILE'])}"
    dataset = create_dataset(model_cfg, dataset_cfg[dset], train=False)

    predictions_list = []
    for ii, batch in enumerate(tqdm(dataset)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = prohmr_model(batch)

        pred_cam_t_full = cam_crop_to_full(
            out['pred_cam'][:, 0],
            batch['box_center'].float(),
            batch['box_size'].float(),
            batch['img_size'].float(),
        ).cpu().numpy()

        temp_betas_ = out['pred_smpl_params']['betas'][:, 0].cpu().numpy()
        temp_pose_ = torch.cat((out['pred_smpl_params']['global_orient'][:, 0],
                                out['pred_smpl_params']['body_pose'][:, 0]), axis=1).cpu().numpy()
        temp_pred_cam_ = out['pred_cam'].cpu().numpy()
        temp_pred_cam_t_ = cam_crop_to_full(
                out['pred_cam'][:, 0],
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

