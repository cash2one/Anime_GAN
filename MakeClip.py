import numpy as np
import cv2
from PIL import Image
for i in range(13,14):
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
        #print(NUM_FPS)

        if(NUM_FPS%30 == 0 and time > 22*60 and time <=44*60):
            frame_color = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame_color = np.array(frame_color)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            frame_gray = np.array(frame_gray)
            #print(frame)
            image = Image.fromarray(frame_color.astype('uint8'), 'RGB')
            image2 = Image.fromarray(frame_gray.astype('uint8'),'L')
            image.save('out_dataset/o_'+str(i)+'_'+str(int(NUM_FPS/30))+'.png')
            image2.save('in_dataset/i_'+str(i)+'_'+str(int(NUM_FPS/30))+'.png')
            #image2.save('dataset_normal/'+str(i)+'_'+str(int(NUM_FPS/30))+'.png')

        # The show ends with keyboard input 'q'
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()