# GAN-based Dataset Augmentor - Mini Project: Semi-Supervised Augmentation

## Overview

This project implements a complete **semi-supervised learning pipeline** for **dataset augmentation** using a **Deep Convolutional GAN (DCGAN)**. The core concept is to generate high-fidelity synthetic images and then leverage a **trained classifier** to assign **pseudo-labels** with high confidence, effectively expanding the training dataset and improving model generalization.

**Key Goals:**
1. **GAN Generation:** Train a DCGAN on the base dataset (e.g., CIFAR-10).  
2. **Pseudo-labeling:** Use a baseline classifier to filter and label generated images based on a strict confidence threshold (e.g., ≥ 0.95).  
3. **Augmentation:** Combine the original dataset with the new pseudo-labeled images to retrain a robust classifier.  
4. **Modular Design:** Provide distinct, reusable Python scripts for each step of the pipeline.

---

## Folder Structure
```markdown
BDML_Mini-Project/
│
├── data/ # Original dataset (MNIST, CIFAR-10)
├── outputs/
│ ├── generated/ # Unlabeled GAN-generated images (e.g., 5000 samples)
│ └── pseudo_labeled/ # Filtered, pseudo-labeled images saved by class (Augmentation data)
├── models/
│ ├── baseline/ # Baseline classifier weights (best_classifier.pth)
│ └── augmented/ # Augmented classifier weights
├── utils/
│ ├── pseudo_label.py # Script for pseudo-labeling and filtering
│ └── ... # Other helper scripts
├── train_dcgan.py # DCGAN training script
├── generate_samples.py # Synthetic image generation script
├── train_classifier.py # General classifier training script
└── README.md
```


## Setup Instructions

### 1. Clone and Enter

```bash
git clone <your-repo-url>
cd BDML_Mini-Project
```
### 2. Set up Python Environment
It's recommended to use a dedicated environment for dependency management.

```bash
conda create -n ml python=3.10
conda activate ml
pip install torch torchvision matplotlib pillow
```
### 3. Dataset Preparation
The project currently supports MNIST and CIFAR-10. Scripts automatically download these datasets into the ./data/ folder upon first run. For custom datasets:

* Ensure your data is properly formatted for PyTorch ImageFolder.

* Adjust the `--nc` (number of channels) and `--img_size` arguments accordingly.

## Project Workflow: Detailed Steps
### Step 1: Train the DCGAN (Generator)
Train the DCGAN to learn the data distribution. This step is only necessary if you don't already have a pre-trained generator.

```bash
python train_dcgan.py \
    --dataset CIFAR10 \
    --epochs 25 \
    --batch_size 128 \
    --img_size 64 \
    --out ./outputs
```
**Arguments:**

* `--dataset` : Dataset to train on (MNIST or CIFAR10)

* `--epochs` : Number of training epochs for the GAN

* `--batch_size` : Training batch size

* `--img_size` : Spatial size of the generated images

* `--out` : Output directory where generator weights (dcgan_generator.pth) are saved

### Step 2: Generate Synthetic Images
Use the saved generator model to produce a large, fixed quantity of unlabeled synthetic images.

```bash
python generate_samples.py \
    --gen_path ./outputs/dcgan_generator.pth \
    --out ./outputs/generated \
    --num_samples 5000
```
**Arguments:**

* `--gen_path` : Path to the trained generator model weights

* `--out` : Output folder for saving the generated images

`--num_samples` : Total number of images to generate

### Step 3: Train the Baseline Classifier
Train a simple CNN classifier on the original labeled dataset. This model will serve as the pseudo-labeler.

```bash
python train_classifier.py \
    --dataset CIFAR10 \
    --epochs 10 \
    --img_size 64 \
    --out ./models/baseline
```
**Output:** Best performing weights are saved to ./models/baseline/best_classifier.pth.

### Step 4: Pseudo-label Synthetic Images
Use the trained baseline classifier to predict labels for the images in ./outputs/generated. Only images where prediction confidence exceeds the specified threshold (≥ 0.95) are saved.

```bash
python utils/pseudo_label.py \
    --synth_dir ./outputs/generated \
    --model_path ./models/baseline/best_classifier.pth \
    --out_dir ./outputs/pseudo_labeled \
    --conf_thresh 0.95
```
**Arguments:**

* `--synth_dir` : Input folder containing the unlabeled GAN-generated images

* `--model_path` : Path to the trained baseline classifier weights

* `--out_dir` : Output folder where filtered images are saved by predicted class

* `--conf_thresh` : Confidence threshold (0–1). Only predictions above this value are kept

### Step 5: Train Classifier on Augmented Dataset
Retrain the classifier using the original labeled data combined with the new, high-confidence pseudo-labeled data.

```bash
python train_classifier.py \
    --dataset CIFAR10 \
    --synthetic_dir ./outputs/pseudo_labeled \
    --epochs 10 \
    --img_size 64 \
    --out ./models/augmented
```
**Arguments:**

* `--synthetic_dir` : Folder containing the class-wise pseudo-labeled images for augmentation

* `--out` : Output folder for the final, augmented model weights

## Notes & Tips
**Filtering**: If too few GAN images are pseudo-labeled, try lowering the --conf_thresh (e.g., from 0.95 to 0.85).

**Modularity**: You can swap out the contents of ./outputs/generated with any unlabeled images (from a different GAN) and rerun the pseudo-labeling script.

**Next Steps**: The augmented classifier in ./models/augmented/ is ready for deployment or further evaluation against the baseline model.

## References

* PyTorch Documentation
* Torchvision Datasets
* DCGAN / CNN architectures adapted for mini-project use.