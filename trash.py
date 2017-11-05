import os
from multiprocessing import Queue
from Train_manager2 import Kill_Zombie_Process
import time
"""
datalist = [x for x in os.listdir('checkpoint/small_data/')]
for x in datalist:
    y = x.rstrip('.pth')
    y = y.lstrip('checkpoint/small_data/netG_model_epoch_')
    y = y.lstrip('checkpoint/small_data/netD_model_epoch_')
    if int(y)%50 != 0:
        os.remove('checkpoint/small_data/netG_model_epoch_{}.pth'.format(y))
        print('checkpoint/small_data/netG_model_epoch_{}.pth'.format(y))
        os.remove('checkpoint/small_data/netD_model_epoch_{}.pth'.format(y))
        print('checkpoint/small_data/netD_model_epoch_{}.pth'.format(y))
"""