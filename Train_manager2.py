import os
import psutil
from multiprocessing import Process, Queue, freeze_support
import time
from train import InitNN, train, test, checkpoint, ANIME_GAN
from MakeClip import pick_old_data, random_data

def Kill_Zombie_Process():
    list = [p for p in psutil.process_iter()]
    list_tokill=[]
    for p in list:
        if p.name() == 'python.exe' and p.pid != os.getpid():
            list_tokill.append(p)
    if len(list_tokill)>0:
        print(list_tokill)
        for process in list_tokill:
            print(process.pid)
            os.kill(process.pid, 6)
            print('{} Killed'.format(process.pid))
            print(os.getpid())
    print('ALL Minor Process Terminated')


net_filenames = [x for x in os.listdir('checkpoint/small_data/')]
net_g_model = None
net_d_model = None
Epoch_Num = 0
q = Queue()
if __name__ == '__main__':
    freeze_support()
    p = Process(target=ANIME_GAN,args=(q,))
    p.start()
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
    time_to_reset=1300
    time_to_print=10
    while True:
        #print('USING RAM : {}%'.format(psutil.virtual_memory().percent))
        if not q.empty():
            print(q.get())
            p.terminate()
            while p.exitcode is None:
                time.sleep(0.01)
            Kill_Zombie_Process()
            pick_old_data()
            random_data()
            time_to_reset = 3000+time.clock()
            print("GAN_PAUSED at {}sec".format(time_to_print))
            time.sleep(10)
            p = Process(target=ANIME_GAN, args=(q,))
            p.start()
        if time.clock()>time_to_print:
            time_to_print+=10
            print('Running Time: {}sec'.format(time.clock()))
        if time.clock() > time_to_reset or psutil.virtual_memory().percent > 80:
            time_to_reset += 1300
            p.terminate()
            while p.exitcode is None:
                time.sleep(0.01)
            Kill_Zombie_Process()
            print("GAN_PAUSED at {}sec".format(time_to_print))
            time.sleep(10)
            p = Process(target=ANIME_GAN,args=(q,))
            p.start()