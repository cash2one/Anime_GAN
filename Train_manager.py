import os
import psutil
import subprocess
import time


net_filenames = [x for x in os.listdir('checkpoint/small_data/')]
net_g_model = None
net_d_model = None
Epoch_Num = 0
subprocess.Popen('{} train.py --threads 8'.format(
    'C:\\Users\\JaeyoungJo\\AppData\\Local\\Continuum\\Anaconda3\\envs\\tensorflow\\python.exe'),close_fds=True)

"""
while True:
    print(psutil.virtual_memory().percent)
    if psutil.virtual_memory().percent > 90:
        list = [p for p in psutil.process_iter()]
        for p in list:
            if p.memory_percent() > 20 and p.name() == 'python.exe' and p.pid != os.getpid():
                os.kill(p.pid,6)
                subprocess.Popen
                ('{} trash.py'.format('C:\\Users\\wogns\\AppData\\Local\\conda\\conda\\envs\\tensorflow\\python.exe')
                 ,close_fds=True)
"""
time_to_reset=3000
time_to_print=10
while True:
    list_tokill=[]
    #print('USING RAM : {}%'.format(psutil.virtual_memory().percent))
    if time.clock()>time_to_print:
        time_to_print+=10
        print('Running Time: {}sec'.format(time.clock()))
    if time.clock()>time_to_reset:
        time_to_reset += 3000
        list = [p for p in psutil.process_iter()]
        for p in list:
            if p.name() == 'python.exe' and p.pid != os.getpid():
                list_tokill.append(p)
        if len(list)>0:
            for process in list_tokill:
                print(process.name())
                os.kill(process.pid, 6)
            time.sleep(10)
            subprocess.Popen('{} train.py --threads 8'.format(
                'C:\\Users\\JaeyoungJo\\AppData\\Local\\Continuum\\Anaconda3\\envs\\tensorflow\\python.exe'),
                close_fds=True)
