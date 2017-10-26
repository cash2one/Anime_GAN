#based on https://github.com/mrzhu-cool/pix2pix-pytorch/blob/ba031e6040560c2b817f3cedf5eb40e5a9206ccb/test.py
from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from util import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', type=str, default='small_data', help='facades')
parser.add_argument('--model', type=str, default='checkpoint/small_data/netG_model_epoch_1.pth', help='model file to use')
parser.add_argument('--cuda', action='store_true',default= True, help='use cuda')
opt = parser.parse_args()
print(opt)

netG = torch.load(opt.model)

image_dir = "dataset/{}/test/a/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = Variable(img, volatile=True).view(1,-1,360,640)

    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()

    out = netG(input)
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))