import os
import torch
import argparse
import shutil
import numpy as np
from hmr2.datasets import create_dataset
from hmr2.configs import CACHE_DIR_4DHUMANS
from prohmr.models import ProHMR
from prohmr.configs import prohmr_config
from hmr2.utils.renderer import cam_crop_to_full
from score_hmr.utils import recursive_to
from score_hmr.configs import dataset_config
from tqdm import tqdm


DATASETS = ["3DPW-TEST"]
OUT_DIR = 'cache/prohmr'
os.makedirs(OUT_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference.")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used for data loading.")
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset_cfg = dataset_config()


# Load proHMR.
print("Loading proHMR model...")
model_cfg = prohmr_config()
prohmr_model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg)
prohmr_model = prohmr_model.to(device)
prohmr_model.eval()

for dset in DATASETS:
    betas_ = []
    pose_ = []
    pred_cam_ = []
    pred_cam_t_ = []

    print(f"Generating predictions on {dset}")
    npz_path = f"{OUT_DIR}/{os.path.basename(dataset_cfg[dset]['DATASET_FILE'])}"

    dataset = create_dataset(prohmr_model.cfg, dataset_cfg[dset], train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    for batch in tqdm(dataloader):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            preds = prohmr_model(batch)

        # Extract predictions
        pred_cam_t_full = cam_crop_to_full(
            preds['pred_cam'][:, 0, :],
            batch['box_center'].float(),
            batch['box_size'].float(),
            batch['img_size'].float(),
        ).cpu().numpy()

        betas_.append(preds['pred_smpl_params']['betas'][:, 0].cpu().numpy())
        pose_.append(
            torch.cat((preds['pred_smpl_params']['global_orient'][:, 0],
                       preds['pred_smpl_params']['body_pose'][:, 0]), axis=1).cpu().numpy()
        )
        pred_cam_.append(preds['pred_cam'][:, 0, :].cpu().numpy())
        pred_cam_t_.append(pred_cam_t_full)

    print('Saving npz file...')
    np.savez(
        npz_path,
        pred_betas=np.concatenate(betas_),
        pred_pose=np.concatenate(pose_),
        pred_cam=np.concatenate(pred_cam_),
        pred_cam_t=np.concatenate(pred_cam_t_),
    )
