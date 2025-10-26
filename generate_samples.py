# generate_samples.py
import argparse
import os
import torch
from model.dcgan import Generator
from utils.train_utils import set_seed
from torchvision.utils import save_image
import math

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    gen = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    gen.load_state_dict(torch.load(args.gen_path, map_location=device))
    gen.eval()
    os.makedirs(args.out, exist_ok=True)
    total = args.num_samples
    bs = args.batch_size
    idx = 0
    with torch.no_grad():
        for _ in range(math.ceil(total/bs)):
            current_bs = min(bs, total - idx)
            noise = torch.randn(current_bs, args.nz, 1, 1, device=device)
            fake = gen(noise)
            fake = (fake + 1) / 2.0  # rescale to [0,1]
            for i in range(fake.size(0)):
                save_image(fake[i], os.path.join(args.out, f'fake_{idx+i:06d}.png'))
            idx += current_bs
            if idx >= total:
                break
    print(f"Saved {idx} synthetic images to {args.out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', required=True)
    parser.add_argument('--out', default='./outputs/generated')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate(args)
