import os
import torch

net_filenames = [x for x in os.listdir('checkpoint/small_data/')]
net_g_model = None
net_d_model = None
Epoch_Num = 0
for i in range(1,201):
    if  'netG_Model_epoch_{}.pth'.format(i) in net_filenames:
        net_g_model = torch.load('checkpoint/small_data/netG_Model_epoch_{}.pth'.format(i))
        net_d_model = torch.load('checkpoint/small_data/netD_Model_epoch_{}.pth'.format(i))
        Epoch_Num = i
    else:
        break
