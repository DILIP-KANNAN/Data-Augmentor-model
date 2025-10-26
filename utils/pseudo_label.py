# utils/pseudo_label_unlabeled.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_classifier_retrained import SimpleCNN, get_transform

def pseudo_label_unlabeled(synth_dir, model_path, out_dir, device='cuda', conf_thresh=0.9, img_size=64, nc=3, num_classes=10):
    os.makedirs(out_dir, exist_ok=True)
    transform = get_transform(img_size=img_size, nc=nc)

    # Load model
    model = SimpleCNN(nc=nc, num_classes=num_classes)
    model_path = os.path.abspath(model_path)
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    files = sorted(glob.glob(os.path.join(synth_dir, '*.png')))
    kept = 0
    for f in files:
        img = Image.open(f).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        print(f"{os.path.basename(f)} -> Pred: {pred.item()}, Conf: {conf.item():.4f}")

        if conf.item() >= conf_thresh:
            # Save in class folder according to predicted label
            lbl = pred.item()
            dest = os.path.join(out_dir, str(lbl))
            os.makedirs(dest, exist_ok=True)
            img.save(os.path.join(dest, os.path.basename(f)))
            kept += 1

    print(f"Saved {kept} pseudo-labeled images to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth_dir', required=True, help='Folder with GAN-generated images')
    parser.add_argument('--model_path', required=True, help='Path to trained classifier weights')
    parser.add_argument('--out_dir', default='./outputs/pseudo_labeled', help='Where to save pseudo-labeled images')
    parser.add_argument('--conf_thresh', type=float, default=0.9)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pseudo_label_unlabeled(
        synth_dir=args.synth_dir,
        model_path=args.model_path,
        out_dir=args.out_dir,
        device=device,
        conf_thresh=args.conf_thresh,
        img_size=args.img_size,
        nc=args.nc,
        num_classes=args.num_classes
    )
