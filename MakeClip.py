import numpy as np
import cv2
from PIL import Image
import os
import random
"""
for i in range(5,6):
    cap = cv2.VideoCapture('MadeinAbyss/' +str(int(i/10))+str(i%10)+'.mkv')
    #cap = cv2.VideoCapture('test.mp4')
    ret = True
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(ret!=True):
            break
        #ret means if the movie is on or off
        #frame is W*H*C size tensor with color info.
        # Display the resulting frame
        #cv2.imshow('frame', frame)
        NUM_FPS = cap.get(cv2.CAP_PROP_POS_FRAMES)
        time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        print('{}, {}'.format(NUM_FPS,time))

        if(NUM_FPS%60 == 0 and time > 5*60 and time <=20*60):
            frame_color = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame_color = np.array(frame_color)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            frame_gray = np.array(frame_gray)
            #print(frame)
            image = Image.fromarray(frame_color.astype('uint8'), 'RGB')
            image2 = Image.fromarray(frame_gray.astype('uint8'),'L')
            image.save('dataset/small_data/train/b/'+str(479+int(NUM_FPS/60))+'.jpg')
            image2.save('dataset/small_data/train/a/'+str(479+int(NUM_FPS/60))+'.jpg')
            #image2.save('dataset_normal/'+str(i)+'_'+str(int(NUM_FPS/30))+'.png')

        # The show ends with keyboard input 'q'
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
"""
def new_dataset(i):
    cap = cv2.VideoCapture('MadeinAbyss/' +str(int(i/10))+str(i%10)+'.mkv')
    #cap = cv2.VideoCapture('test.mp4')
    ret = True
    print('{}.mkv is adding.....'.format(i))
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(ret!=True):
            break
        #ret means if the movie is on or off
        #frame is W*H*C size tensor with color info.
        # Display the resulting frame
        #cv2.imshow('frame', frame)
        NUM_FPS = cap.get(cv2.CAP_PROP_POS_FRAMES)
        time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        #print('{}, {}'.format(NUM_FPS,time))
        if(time > 20*60):
            break;
        if(NUM_FPS%60 == 0 and time > 5*60 and time <=20*60):
            frame_color = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame_color = np.array(frame_color)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            frame_gray = np.array(frame_gray)
            #print(frame)
            image = Image.fromarray(frame_color.astype('uint8'), 'RGB')
            image2 = Image.fromarray(frame_gray.astype('uint8'),'L')
            image.save('dataset/small_data/train/b/'+str(500*i+int(NUM_FPS/60))+'.jpg')
            image2.save('dataset/small_data/train/a/'+str(500*i+int(NUM_FPS/60))+'.jpg')
            #image2.save('dataset_normal/'+str(i)+'_'+str(int(NUM_FPS/30))+'.png')

            # The show ends with keyboard input 'q'
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
def pick_old_data():
    list = [x for x in os.listdir('dataset/small_data/train/a')]
    random.shuffle(list)
    for name in list[:360]:
        os.remove('dataset/small_data/train/a/{}'.format(name))
        os.remove('dataset/small_data/train/b/{}'.format(name))
    print('PICK DATASET SUCCESS')
def random_data():
    f = open('datalist.txt', 'r')
    datalist = f.readlines()
    for i in range(len(datalist)):
        datalist[i]=int(datalist[i].strip())
    f.close()
    data = int(random.choice(datalist))
    new_dataset(data)
    datalist.remove(data)
    f = open('datalist.txt', 'w')
    for d in datalist:
        f.write('{}\n'.format(d))
    f.close()

if __name__ == '__main__':
    for i in range(1,14):
        if i is not 2:
            #pick_old_data()
            new_dataset(i)
    #pick_old_data()
    #random_data()