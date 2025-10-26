# evaluate_augmentation.py
import os
import subprocess
import argparse

def run_command(cmd):
    print(f">>> {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--synth_dir', default='./outputs/generated')
    parser.add_argument('--img_size', type=int, default=64)
    args = parser.parse_args()

    # 1) Train baseline classifier
    print("Step 1: Train baseline classifier on original data")
    run_command(f"python train_classifier.py --dataset {args.dataset} --data_root {args.data_root} --epochs 10 --img_size {args.img_size} --out ./models/baseline")

    # 2) Generate synthetic images (assumes generator saved at outputs/dcgan_generator.pth)
    print("Step 2: Generate synthetic images (if not present)")
    if not os.path.exists(args.synth_dir) or len(os.listdir(args.synth_dir)) < 100:
        os.makedirs(args.synth_dir, exist_ok=True)
        run_command(f"python generate_samples.py --gen_path ./outputs/dcgan_generator.pth --out {args.synth_dir} --num_samples 5000 --batch_size 64")

    # 3) Pseudo-label synthetic images using baseline
    print("Step 3: Pseudo-label synthetic images")
    run_command("python -c \"from utils.pseudo_label import pseudo_label; import torch; "
                "device='cuda' if torch.cuda.is_available() else 'cpu'; "
                "pseudo_label(synth_dir='./outputs/generated', model_path='./models/baseline/best_classifier.pth', out_dir='./outputs/pseudo_labeled', device=device, conf_thresh=0.95, img_size=64)\"")

    # 4) Train classifier on augmented dataset
    print("Step 4: Train classifier on original + pseudo-labeled synthetic data")
    run_command(f"python train_classifier.py --dataset {args.dataset} --data_root {args.data_root} --synthetic_dir ./outputs/pseudo_labeled --epochs 10 --img_size {args.img_size} --out ./models/augmented")

    print("Evaluation complete. Check model accuracies in models/baseline and models/augmented")
