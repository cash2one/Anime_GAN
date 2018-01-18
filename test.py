#based on https://github.com/mrzhu-cool/pix2pix-pytorch/blob/ba031e6040560c2b817f3cedf5eb40e5a9206ccb/test.py
from __future__ import print_function
import argparse
import os
import psutil
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from util import is_image_file, load_img, save_img

# Testing settings

#-----test latest NN----------#
def test():
    net_filenames = [x for x in os.listdir('checkpoint/small_data_L2/')]
    EpochNumber = 0
    page_num = EpochNumber
    for j in range(1,50):
        if 'netG_model_epoch_{}.pth'.format(page_num + j) in net_filenames:
            print(page_num + j)
            EpochNumber = page_num + j
        else:
            break
    for i in range(1, 10001):
        if 'netG_model_epoch_{}.pth'.format(i * 50) in net_filenames:
            print(i * 50)
            EpochNumber = i * 50
        else:
            break
    page_num = EpochNumber
    for j in range(50):
        if 'netG_model_epoch_{}.pth'.format(page_num + j) in net_filenames:
            print(page_num + j)
            EpochNumber = page_num + j
        else:
            break
    #EpochNumber = 1

    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--dataset', type=str, default='small_data', help='facades')
    parser.add_argument('--model', type=str, default='checkpoint/small_data_L2/netG_model_epoch_{}.pth'.format(EpochNumber), help='model file to use')
    parser.add_argument('--cuda', action='store_true',default= True, help='use cuda')
    opt = parser.parse_args()
    print(opt)

    netG = torch.load(opt.model)

    image_dir = "dataset/{}/test/a/".format(opt.dataset)
    image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    if not os.path.exists(os.path.join("result", opt.dataset+'_L2', 'Epoch{}'.format(EpochNumber))):
        os.mkdir(os.path.join("result", opt.dataset+'_L2', 'Epoch{}'.format(EpochNumber)))
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

        save_img(out_img, "result/{}_L2/{}/{}".format(opt.dataset, 'Epoch{}'.format(EpochNumber), image_name))

    #-----print out training data result-----#
    image_dir = "dataset/{}/train/a/".format(opt.dataset)
    image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    if not os.path.exists(os.path.join("result", opt.dataset+'_L2', 'Epoch{}_train'.format(EpochNumber))):
        os.mkdir(os.path.join("result", opt.dataset+'_L2', 'Epoch{}_train'.format(EpochNumber)))
    for image_name in image_filenames[:100]:
        img = load_img(image_dir + image_name)
        img = transform(img)
        input = Variable(img, volatile=True).view(1,-1,360,640)

        if opt.cuda:
            netG = netG.cuda()
            input = input.cuda()

        out = netG(input)
        out = out.cpu()
        out_img = out.data[0]

        save_img(out_img, "result/{}_L2/{}/{}".format(opt.dataset, 'Epoch{}_train'.format(EpochNumber), image_name))

if __name__ == '__main__' :
    test()