
import os
import math

import cv2
import glob
import imageio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
from tqdm import tqdm
from os.path import join as pjoin

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from data_loader import prepare_loader


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        nz = opt.latent_dim
        ngf = opt.img_size
        nc = opt.channels
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        # z, shape=(opt.batch_size, opt.latent_dim)
        img = self.main(z)
        # img, shape=(opt.batch_size, opt.channels, opt.img_size, opt.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = opt.channels
        ndf = opt.img_size
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        # img, shape=(opt.batch_size, opt.channels, opt.img_size, opt.img_size)
        validity = self.main(img)
        # validity, shape=(opt.batch_size, 1)
        return validity

def load_checkpoint(generator, discriminator,optimizer_G, optimizer_D, filename, device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    #generator, discriminator, adversarial_loss, optimizer_G, optimizer_D
    start_epoch = 0
    iters = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device)
        generator.load_state_dict(checkpoint['netG_state_dict'])
        discriminator.load_state_dict(checkpoint['netD_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return generator, discriminator #, adversarial_loss, optimizer_G, optimizer_D



def train(opt):
    os.makedirs("result/images_train", exist_ok=True)
    os.makedirs("result/models", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


    # Config dataloader
    class DogDataset(torch.utils.data.Dataset):
        def __init__(self, path):
            super().__init__()

            #self.images = None
            self.images = []

            # Read all png files
            for im_path in tqdm(glob.glob(pjoin(path, "*.png"))):
                im = imageio.imread(im_path)
                im = np.array(im, dtype='float')
                self.images.append(im)

            self.images = np.array(self.images)
            self.transform = A.Compose(
                [A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 ToTensor()])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.images[idx]
            img = self.transform(image=img)['image']

            return img

    dataloader = torch.utils.data.DataLoader(DogDataset('data/images_crop'),
                                             batch_size=opt.batch_size,
                                             shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.lr,
                                   betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=opt.lr,
                                   betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    checkpoint = torch.load('generator.pt', map_location='cuda')
    generator.load_state_dict(checkpoint)
    checkpoint = torch.load('discriminator.pt', map_location='cuda')
    discriminator.load_state_dict(checkpoint)
    generator.cuda()
    discriminator.cuda()
    
    # now individually transfer the optimizer parts...
    for state in optimizer_G.state.values():
        for k, v in state.items():
            if isinstance(v, Tensor):
                state[k] = v.cuda()

    # now individually transfer the optimizer parts...
    for state in optimizer_D.state.values():
        for k, v in state.items():
            if isinstance(v, Tensor):
                state[k] = v.cuda()
    generator.train()
    discriminator.train()
    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0),
                             requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0),
                            requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 1,
                                        (imgs.shape[0], opt.latent_dim,1,1))))

            # Generate a batch of images
            gen_imgs = generator(z)
            output = discriminator(gen_imgs).view(-1,1)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(output,valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            output = discriminator(real_imgs).view(-1,1)
            real_loss = adversarial_loss(output,valid)
            #real_loss.backward()
            
            gen_imgs = generator(z)
            output = discriminator(gen_imgs).view(-1,1)
            fake_loss = adversarial_loss(output,fake)
            #fake_loss.backward()

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            '''print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                  (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(),
                   g_loss.item()))'''

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25],
                           "result/images_train/%d.png" % batches_done,
                           nrow=5,
                           normalize=True)
                os.makedirs("result/models/%d" % batches_done, exist_ok=True)
                torch.save(discriminator.state_dict(),
                           "result/models/%d/discriminator.pt" % batches_done)
                torch.save(generator.state_dict(),
                           "result/models/%d/generator.pt" % batches_done)
                torch.save(optimizer_G.state_dict(),
                           "result/models/%d/optimizer_G.pt" % batches_done)
                torch.save(optimizer_D.state_dict(),
                           "result/models/%d/optimizer_G.pt" % batches_done)
        print("[Epoch %d/%d] [D loss: %f ] [G loss: %f ]" %
                  (epoch+1, opt.n_epochs, d_loss.item(),g_loss.item()))

def inference(opt):
    os.makedirs("result/images_inference", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Load trained models
    discriminator.load_state_dict(
        torch.load(opt.model_path + '/discriminator.pt'))
    generator.load_state_dict(torch.load(opt.model_path + '/generator.pt'))

    # Inference
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    z = Variable(
        Tensor(np.random.normal(0, 1, (opt.inference_num, opt.latent_dim,1,1))))
    gen_imgs = generator(z)

    for i in tqdm(range(0, opt.inference_num)):
        save_image(gen_imgs.data[i],
                   "result/images_inference/%d.png" % i,
                   normalize=True)


def process_data(opt):
    """ Processes data according to bbox in annotations. """

    os.makedirs("data/images_crop", exist_ok=True)
    os.makedirs("data/images_ref", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    # Config dataloader
    root_images = pjoin(opt.data_path, 'images/') #'data/images/'
    root_annots = pjoin(opt.data_path, 'annotations/')#'data/annotations/'
    dataloader = prepare_loader(root_images, root_annots, opt.batch_size)

    # Save cropped images
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    image_count = 0
    for imgs  in tqdm(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        for i in range(0, len(real_imgs)):
            save_image(real_imgs.data[i],
                       "data/images_crop/%d.png" % image_count,
                       normalize=True)

            # Save first K images as reference for calculating FID scores.
            if image_count < opt.inference_num:
                save_image(real_imgs.data[i],
                           "data/images_ref/%d.png" % image_count,
                           normalize=True)
            image_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=['train', 'inference', 'process_data'],
                        required=True,
                        help="operation mode")
    parser.add_argument("--model_path",
                        type=str,
                        default='models/0',
                        help="model path for inference")
    parser.add_argument("--data_path",
                        type=str,
                        default='data',
                        help="data path for data process and training")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=[ADJUST],
                        help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=[ADJUST],
                        help="size of the batches")
    parser.add_argument("--lr",
                        type=float,
                        default=[ADJUST],
                        help="adam: learning rate")
    parser.add_argument("--b1",
                        type=float,
                        default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2",
                        type=float,
                        default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim",
                        type=int,
                        default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size",
                        type=int,
                        default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels",
                        type=int,
                        default=3,
                        help="number of image channels")
    parser.add_argument("--sample_interval",
                        type=int,
                        default=400,
                        help="interval between image sampling")
    parser.add_argument("--inference_num",
                        type=int,
                        default=100,
                        help="number of generated images for inference")
    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'inference':
        inference(opt)
    else:
        process_data(opt)
