# train_dcgan.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from model.dcgan import Generator, Discriminator
from utils.datasets import get_dataloader
import argparse
import numpy as np
from utils.train_utils import set_seed, weights_init, sample_noise

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Create output folders
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)

    # Data
    dataloader = get_dataloader(dataset_name=args.dataset, root=args.data_root,
                                batch_size=args.batch_size, img_size=args.img_size, nc=args.nc)

    # Models
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    netD = Discriminator(nc=args.nc, ndf=args.ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = sample_noise(args.nz, 64, device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    iters = 0
    for epoch in range(args.epochs):
        for i, (data, _) in enumerate(dataloader):
            ############################
            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # Update Generator: maximize log(D(G(z)))
            ############################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if iters % args.print_every == 0:
                print(f"Epoch[{epoch}/{args.epochs}] Iter[{i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

            if iters % args.sample_every == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    grid = make_grid(fake, padding=2, normalize=True)
                    save_image(grid, os.path.join(args.out, 'samples', f'sample_{iters}.png'))

            iters += 1

        # Save checkpoints per epoch
        torch.save(netG.state_dict(), os.path.join(args.out, f'generator_epoch_{epoch}.pth'))
        torch.save(netD.state_dict(), os.path.join(args.out, f'discriminator_epoch_{epoch}.pth'))

    # Save final
    torch.save(netG.state_dict(), os.path.join(args.out, 'dcgan_generator.pth'))
    torch.save(netD.state_dict(), os.path.join(args.out, 'dcgan_discriminator.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST','CIFAR10'])
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--out', default='./outputs')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=500)
    args = parser.parse_args()
    train(args)
