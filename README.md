# Transferring Learning Trajectories of Neural Networks
Replication of Chijiwa et al. (2023) — arXiv:2305.14122v2

pip install -r requirements.txt

## Dataset Setup

**MNIST, CIFAR-10, CIFAR-100** — downloaded automatically by the notebook.

**CUB-200-2011** — downloaded automatically by the notebook. If it fails, download manually from http://www.vision.caltech.edu/datasets/cub_200_2011/ and extract to `data/cub200/`.

**Stanford Cars** — the official Stanford/torchvision URL is permanently broken. Download manually from Kaggle: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset and structure as follows:
```
data/cars/stanford_cars/
├── cars_train/     # training images
├── cars_test/      # test images (unused in this replication)
└── devkit/
    ├── cars_train_annos.mat
    ├── cars_test_annos.mat
    └── cars_meta.mat
```

**ImageNet (ILSVRC 2012)** — must be downloaded manually from https://image-net.org (registration required). Structure as follows:
```
data/imagenet/
├── train/
│   └── nXXXXXXXX/   # one folder per synset
└── val/
    └── nXXXXXXXX/   # one folder per synset
```
The data folder can also be more easily downloaded to my drive via the link: https://drive.google.com/file/d/1-Kd7dV5MVRMCjVLJwmQnJvad3Hyskdy9/view?usp=drive_link (22 Go)
Once downloaded, it should be extracted and placed in the root directory


## Running the Experiments

Open and run `main.ipynb` cell by cell in order.

> ResNet-18 experiments are computationally heavy on CPU (1-2 days per cell).
> A swap partition of at least 16 GB is recommended to avoid out-of-memory crashes:
> ```bash
> sudo fallocate -l 16G /swapfile
> sudo chmod 600 /swapfile
> sudo mkswap /swapfile
> sudo swapon /swapfile

For Experiment 4b Methods 3 and 4, run the standalone scripts instead of the notebook cells to avoid kernel crashes:
```bash
python run_exp4b.py
python run_exp4b_method4.py
```
