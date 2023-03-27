import click
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from data import *
from models import *


#os.makedirs("images", exist_ok=True)
# Define image directories
img_train_dir = "/zhome/0f/f/137116/Desktop/TorturedRats/DRIVE/training/images/"
img_test_dir = "/zhome/0f/f/137116/Desktop/TorturedRats/DRIVE/test/images/"


@click.command()
@click.option('--n_epochs', '-n_epochs', type=click.INT, default=200, help='number of epochs of training')
@click.option('--batch_size', '-batch_size', type=click.INT, default=8, help='size of the batches')
@click.option('--lr', '-lr', type=click.FLOAT, default=1e-4, help='Learning rate, defaults to 1e-4')
@click.option('--b1', '-b1', type=click.FLOAT, default=0.5, help="adam: decay of first order momentum of gradient")
@click.option('--b2', '-b2', type=click.FLOAT, default=0.999, help="adam: decay of first order momentum of gradient")
@click.option('--n_cpu', '-n_cpu', type=click.INT, default=4, help="number of cpu threads to use during batch generation")
@click.option('--latent_dim', '-latent_dim', type=click.INT, default=100, help="dimensionality of the latent space")
@click.option('--img_size', '-img_size', type=click.INT, default=128, help="size of each image dimension")
@click.option('--mask_size', '-mask_size', type=click.INT, default=64, help="size of random mask")
@click.option('--channels', '-channels', type=click.INT, default=3, help="number of image channels")
@click.option('--sample_interval', '-sample_interval', type=click.INT, default=500, help="interval between image sampling")

def main(n_epochs, batch_size, lr, b1, b2, n_cpu, latent_dim, img_size, mask_size, channels, sample_interval):
  
  # Check if CUDA is available
  cuda = True if torch.cuda.is_available() else False

  # Calculate output of image discriminator (PatchGAN)
  patch_h, patch_w = int(mask_size / 2 ** 3), int(mask_size / 2 ** 3)
  patch = (1, patch_h, patch_w)
        
  def weights_init_normal(m):
      classname = m.__class__.__name__
      if classname.find("Conv") != -1:
          torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
      elif classname.find("BatchNorm2d") != -1:
          torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
          torch.nn.init.constant_(m.bias.data, 0.0)

  # Loss function
  adversarial_loss = torch.nn.MSELoss()
  reconstruction_loss = torch.nn.L1Loss()

  # Initialize generator and discriminator
  generator = Generator(channels=channels)
  discriminator = Discriminator(channels=channels)

  # Initialize weights
  generator.apply(weights_init_normal)
  discriminator.apply(weights_init_normal)

  # If cuda is available use cuda
  if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    reconstruction_loss.cuda()
    
  # Define transforms
  transforms_ = [
      transforms.Resize((img_size, img_size), Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ]

  dataloader = DataLoader(
      ImageDataset(img_train_dir, transforms_=transforms_),
      batch_size=batch_size,
      shuffle=True,
      num_workers=n_cpu,
  )
  test_dataloader = DataLoader(
      ImageDataset(img_test_dir, transforms_=transforms_, mode="val"),
      batch_size=12,
      shuffle=True,
      num_workers=1,
  )

  # Optimizers
  optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
  optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

  # Define Tensor type
  Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


  def save_sample(batches_done):
      samples, masked_samples, i = next(iter(test_dataloader))
      samples = Variable(samples.type(Tensor))
      masked_samples = Variable(masked_samples.type(Tensor))
      i = i[0].item()  # Upper-left coordinate of mask
      # Generate inpainted image
      gen_mask = generator(masked_samples)
      filled_samples = masked_samples.clone()
      filled_samples[:, :, i : i + mask_size, i : i + mask_size] = gen_mask
      # Save sample
      sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
      save_image(sample, "/zhome/0f/f/137116/Desktop/TorturedRats/reports/figures/retinalVessel/generative/%d.png" % batches_done, nrow=6, normalize=True)

  # ----------
  #  Training
  # ----------
  print("\n", 8*"-", "Training Starts", 8*"-", "\n")
  print(f' Number of epochs:       {n_epochs}',"\n",
        f'Batch size:             {batch_size}',"\n",
        f'Learning rate:          {lr}',"\n",
        f'Number of cpu threads:  {n_cpu}',"\n",
        f'Image size:             {img_size}',"\n",
        f'Mask size:              {mask_size}',"\n",
        f'Number of channels:     {channels}',"\n",
        f'Sample interval:        {sample_interval}',"\n")
  
  for epoch in range(n_epochs):
      for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):

          # Adversarial ground truths
          valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
          fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

          # Configure input
          imgs = Variable(imgs.type(Tensor))
          masked_imgs = Variable(masked_imgs.type(Tensor))
          masked_parts = Variable(masked_parts.type(Tensor))

          # -----------------
          #  Train Generator
          # -----------------

          optimizer_G.zero_grad()

          # Generate a batch of images
          gen_parts = generator(masked_imgs)

          # Adversarial and reconstruction loss
          g_adv = adversarial_loss(discriminator(gen_parts), valid)
          g_recon = reconstruction_loss(gen_parts, masked_parts)
          # Total loss
          g_loss = 0.001 * g_adv + 0.999 * g_recon

          g_loss.backward()
          optimizer_G.step()

          # ---------------------
          #  Train Discriminator
          # ---------------------

          optimizer_D.zero_grad()

          # Measure discriminator's ability to classify real from generated samples
          real_loss = adversarial_loss(discriminator(masked_parts), valid)
          fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
          d_loss = 0.5 * (real_loss + fake_loss)

          d_loss.backward()
          optimizer_D.step()

          print(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
              % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_recon.item())
          )

          # Generate sample at sample interval
          batches_done = epoch * len(dataloader) + i
          if batches_done % sample_interval == 0:
              save_sample(batches_done)
              
if __name__ == "__main__":
  main()