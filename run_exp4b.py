import os
import sys
import gc
import numpy as np
import yaml

sys.path.insert(0, os.getcwd())

with open('configs/hyperparams.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_ROOT = os.path.join(os.getcwd(), 'data')
RESULTS_DIR = 'results'
CHECKPOINTS_DIR = 'checkpoints'

from experiments.exp4_inheritance import run_method3_fgmt_10pct_to_full

ckpt_10pct = os.path.join(CHECKPOINTS_DIR, 'pretrained', 'imagenet_10pct_resnet18.pth')
ckpt_full  = os.path.join(CHECKPOINTS_DIR, 'pretrained', 'imagenet_full_resnet18.pth')

if os.path.exists(ckpt_10pct) and os.path.exists(ckpt_full):
    print('=== Method 3: FGMT (10% -> Full) ===')
    gc.collect()
    method3_results = run_method3_fgmt_10pct_to_full(
        config=config,
        data_root=DATA_ROOT,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        device='cpu'
    )
    gc.collect()
    np.save(os.path.join(RESULTS_DIR, 'exp4b_method3_results.npy'), method3_results, allow_pickle=True)
    print('Method 3 completed.')
else:
    print('Checkpoints missing.')