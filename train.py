from __future__ import print_function
import argparse
import os
from math import log10
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn

# Training settings
opt = None
#if __name__ == '__main__':
def InitNN():
    #PID = os.getpid()
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    #parser.add_argument('--dataset', required=True, help='facades')
    parser.add_argument('--dataset', type=str, default='small_data', help='facades')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    #parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    opt = parser.parse_args()

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    root_path = "dataset/"
    train_set = get_training_set(root_path + opt.dataset)
    test_set = get_test_set(root_path + opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)

    print('===> Building model')
    #------------------------------Code to continue training of the latest NN--------------------------#
    net_filenames = [x for x in os.listdir('checkpoint/small_data2/')]
    Epoch_Num = 0
    page_num = Epoch_Num
    for j in range(1,50):
        if 'netG_model_epoch_{}.pth'.format(page_num + j) in net_filenames:
            netG = torch.load('checkpoint/small_data2/netG_model_epoch_{}.pth'.format(page_num + j))
            netD = torch.load('checkpoint/small_data2/netD_model_epoch_{}.pth'.format(page_num + j))
            print(page_num + j)
            Epoch_Num = page_num + j
        else:
            break
    for i in range(1,10001):
        if 'netG_model_epoch_{}.pth'.format(i*50) in net_filenames:
            netG = torch.load('checkpoint/small_data2/netG_model_epoch_{}.pth'.format(i*50))
            netD = torch.load('checkpoint/small_data2/netD_model_epoch_{}.pth'.format(i*50))
            print(i*50)
            Epoch_Num = i*50
        else:
            break
    page_num = Epoch_Num
    for j in range(50):
        if 'netG_model_epoch_{}.pth'.format(page_num+j) in net_filenames:
            netG = torch.load('checkpoint/small_data2/netG_model_epoch_{}.pth'.format(page_num+j))
            netD = torch.load('checkpoint/small_data2/netD_model_epoch_{}.pth'.format(page_num+j))
            print(page_num+j)
            Epoch_Num = page_num+j
        else:
            break
    #-------------------------------------------------------------------------------------------------#
    if Epoch_Num == 0:
        netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, [0])
        netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, [0])

    criterionGAN = GANLoss()
    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print('---------- Networks initialized -------------')
    print_network(netG)
    print_network(netD)
    print('-----------------------------------------------')

    real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 360,640)
    #print(real_a)
    real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 360, 640)

    if opt.cuda:
        netD = netD.cuda()
        netG = netG.cuda()
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)
    return Epoch_Num, training_data_loader, testing_data_loader, optimizerG, optimizerD\
        ,netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt


def train(epoch, training_data_loader, testing_data_loader, optimizerG, optimizerD\
        ,netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt):
    D_LOSS_THRESHOLD=0.001
    D_LOSS_SUM=0
    D_LOSS_COUNT=0
    COUNT=0
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        fake_b = netG(real_a)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()

        optimizerG.step()
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.data[0], loss_g.data[0]))
        # ----code for auto updating dataset-----#
        D_LOSS_COUNT +=1
        D_LOSS_SUM += loss_d.data[0]
        print("AVG_D_LOSS:{}".format(D_LOSS_SUM/D_LOSS_COUNT))


    CH_DATA = (D_LOSS_SUM/D_LOSS_COUNT < D_LOSS_THRESHOLD)
    # ---------------------------------------#

    return CH_DATA


def test(training_data_loader, testing_data_loader, optimizerG, optimizerD\
        ,netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt):
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = netG(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))



def checkpoint(epoch, training_data_loader, testing_data_loader, optimizerG, optimizerD\
        ,netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}2/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
    net_d_model_out_path = "checkpoint/{}2/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}/{}2".format("checkpoint", opt.dataset))



#if __name__ == '__main__':
def ANIME_GAN(q):
    Epoch_Num, training_data_loader, testing_data_loader, optimizerG, optimizerD \
        , netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt \
        = InitNN()
    for epoch in range(Epoch_Num+1, opt.nEpochs + 1):
        #send error message by try-except
        CH_DATA = \
            train(epoch, training_data_loader, testing_data_loader, optimizerG, optimizerD
              , netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt)
        #test(training_data_loader, testing_data_loader, optimizerG, optimizerD\
            #,netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b)
        #if epoch % 50 == 0:
        checkpoint(epoch, training_data_loader, testing_data_loader, optimizerG, optimizerD
                   , netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt)
        print("Epoch {} Finished".format(epoch))
        if epoch % 50 == 0:
            for i in range(1,50):
                os.remove("checkpoint/{}2/netG_model_epoch_{}.pth".format(opt.dataset, epoch-i))
                os.remove("checkpoint/{}2/netD_model_epoch_{}.pth".format(opt.dataset, epoch - i))
        if epoch % 100 == 0:
            print('SEND MESSAGE to manager from trainer')
            q.put('CHANGE_DATASET')
            print('SENT MESSAGE to manager from trainer SUCCESS')
    return Epoch_Num, training_data_loader, testing_data_loader, optimizerG, optimizerD \
        , netG, netD, criterionMSE, criterionL1, criterionGAN, real_a, real_b, opt

if __name__ == '__main__':
    freeze_support()
    ANIME_GAN()