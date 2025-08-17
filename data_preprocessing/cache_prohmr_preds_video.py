import os
import torch
import argparse
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


DATASETS = ["3DPW-TEST-VIDEO"]
OUT_DIR = 'cache/prohmr'
os.makedirs(OUT_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    betas_ = []
    pose_ = []
    pred_cam_ = []
    pred_cam_t_ = []

    print(f"Generating predictions on {dset}")
    # Dataset.
    dataset_cfg = dataset_config()[dset]
    pkl_path = f"{OUT_DIR}/{os.path.basename(dataset_cfg['DATASET_FILE'])}"
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)

    predictions_list = []
    for batch in tqdm(dataset):
        batch_size = 100
        batch_slice = {}

        temp_betas_ = []
        temp_pose_ = []
        temp_pred_cam_ = []
        temp_pred_cam_t_ = []

        for i in range(0, batch['img'].shape[0], batch_size):
            end_i = min(i + batch_size, batch['img'].shape[0])

            batch_slice['img'] = batch['img'][i:end_i]
            batch_slice['keypoints_2d'] = batch['keypoints_2d'][i:end_i]
            batch_slice['keypoints_3d'] = batch['keypoints_3d'][i:end_i]
            batch_slice['smpl_params'] = {
                'global_orient': batch['smpl_params']['global_orient'][i:end_i],
                'body_pose': batch['smpl_params']['body_pose'][i:end_i],
                'betas': batch['smpl_params']['betas'][i:end_i],
            }
            batch_slice['has_smpl_params'] = {
                'global_orient': batch['has_smpl_params']['global_orient'][i:end_i],
                'body_pose': batch['has_smpl_params']['body_pose'][i:end_i],
                'betas': batch['has_smpl_params']['betas'][i:end_i],
            }
            batch_slice['smpl_params_is_axis_angle'] = {
                'global_orient': batch['smpl_params_is_axis_angle']['global_orient'][i:end_i],
                'body_pose': batch['smpl_params_is_axis_angle']['body_pose'][i:end_i],
                'betas': batch['smpl_params_is_axis_angle']['betas'][i:end_i],
            }
            batch_slice['imgname'] = batch['imgname'][i:end_i]
            batch_slice['box_center'] = batch['box_center'][i:end_i]
            batch_slice['box_size'] = batch['box_size'][i:end_i]
            batch_slice['img_size'] = batch['img_size'][i:end_i]
            batch_slice['pred_betas'] = batch['pred_betas'][i:end_i]
            batch_slice['pred_pose'] = batch['pred_pose'][i:end_i]
            batch_slice['pred_cam'] = batch['pred_cam'][i:end_i]
            batch_slice['pred_cam_t'] = batch['pred_cam_t'][i:end_i]
            batch_slice['orig_keypoints_2d'] = batch['orig_keypoints_2d'][i:end_i]

            batch_slice = recursive_to(batch_slice, device)
            with torch.no_grad():
                preds = prohmr_model(batch_slice)

            temp_betas_.append(preds['pred_smpl_params']['betas'][:, 0].cpu().numpy())
            temp_pose_.append(
                torch.cat((preds['pred_smpl_params']['global_orient'][:, 0],
                           preds['pred_smpl_params']['body_pose'][:, 0]), axis=1).cpu().numpy()
            )
            temp_pred_cam_.append(preds['pred_cam'][:, 0, :].cpu().numpy())
            temp_pred_cam_t_.append(
                cam_crop_to_full(
                    preds['pred_cam'][:, 0, :],
                    batch_slice['box_center'].float(),
                    batch_slice['box_size'].float(),
                    batch_slice['img_size'].float(),
                ).cpu().numpy()
            )

        # Extract predictions
        # betas_.append(np.concatenate(temp_betas_, axis=0))
        # pose_.append(np.concatenate(temp_pose_, axis=0))
        # pred_cam_.append(np.concatenate(temp_pred_cam_, axis=0))
        # pred_cam_t_.append(np.concatenate(temp_pred_cam_t_, axis=0))

        # Combine all predictions into a dictionary
        predictions = {
            'pred_betas': np.concatenate(temp_betas_),
            'pred_pose': np.concatenate(temp_pose_),
            'pred_cam': np.concatenate(temp_pred_cam_),
            'pred_cam_t': np.concatenate(temp_pred_cam_t_),
        }

        predictions_list.append(predictions)

    print('Saving pkl file...')
    # Save results to .pkl file
    with open(pkl_path, 'wb') as f:
        pickle.dump(predictions_list, f)

print(f"Results saved to {pkl_path}")
