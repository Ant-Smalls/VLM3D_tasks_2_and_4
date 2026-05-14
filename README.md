# VLM3D Challenge - Task 2: Multi-Abnormality Classification

This repository contains the training and evaluation code for Task 2 of the VLM3D Challenge, focusing on multi-abnormality classification from 3D medical imaging (NIfTI format). The pipeline leverages PyTorch Lightning, MONAI, and pre-trained vision transformers (e.g., DINOv2, DINOv3) adapted for 3D volumetric data.

## Project Structure

```text
task2_classification_challenge/
├── src/
│   ├── modules/               # Core model components (dataloaders, transforms, encoders, classification heads)
│   ├── pretrained_encoders/   # Local weights for DINOv2 and DINOv3 models
│   ├── scripts/
│   |   ├── download_data/     # Scripts to download dataset and labels
│   |   ├── subset_data/       # Utilities for dataset subsetting and exploration
│   |   ├── train.py           # Main training script
│   |   └── test.py            # Main testing and evaluation script 
|   |-- sampled_train_fixed.   # example folder for placing the data files; subset_manifest.json is located here
├── requirements.txt           # Python dependencies
├── slurm_script.txt           # Reference SLURM job submission scripts
└── README.md                  # Project documentation
```

## Setup & Installation

1. **Clone the repository and navigate to the project directory:**
  ```bash
   git clone https://github.com/Ant-Smalls/VLM3D_tasks_2_and_4
   cd task2_classification_challenge
  ```
2. **Create and activate a virtual environment (recommended):**
  ```bash
   python3 -m venv venv
   source venv/bin/activate
  ```
3. **Install the required dependencies:**
  ```bash
   pip install -r requirements.txt
  ```

## Usage

### 1. Data Preparation

The dataset consists of 3D NIfTI images (`.nii`) and a labels CSV file. Use the scripts in `src/scripts/download_data/` to fetch the data. The data loader automatically handles creating train/validation/test splits via a subset manifest JSON if one does not exist.

### 2. Training the Model

You can train the model using `train.py`. The script accepts various hyperparameters for the dataloader, model, and PyTorch Lightning trainer. 

**Local Run:**

```bash
python3 src/scripts/train.py \
    --data_dir ./path/to/data \
    --labels_csv src/scripts/download_data/training_data_labels/train_labels.csv \
    --encoder_dir src/pretrained_encoders/dinov2-small \
    --output_dir src/outputs \
    --max_epochs 20 \
    --batch_size 4 \
    --encoder_type dinov2
```

**Cluster (SLURM) Run:**
A reference SLURM script is provided in `slurm_script.txt`. It requests a GPU node and runs the PyTorch Lightning training loop.

### 3. Testing and Evaluation

Once training is complete, the best checkpoint based on validation Mean AUC is saved in `src/outputs/checkpoints/`. Use `test.py` to evaluate your model on the test split.

```bash
python3 src/scripts/test.py \
    --checkpoint src/outputs/checkpoints/best-epoch=11-val_mean_auc=0.6067.ckpt \
    --data_dir ./path/to/data \
    --batch_size 4 \
    --num_workers 2
```

## Key Technologies

- **PyTorch & PyTorch Lightning:** Model building and training loop management.
- **MONAI:** Medical imaging-specific data loading and 3D augmentations/transforms.
- **Hugging Face Transformers:** Integration of pre-trained vision encoders.
- **TensorBoard:** Logging of training metrics and losses.

