import numpy as np
import cv2
from PIL import Image
import os

image_names=[x for x in os.listdir('result/small_data/Epoch18')]
"""
for names in image_names:
    print('result/small_data/Epoch18/{}'.format(names))
    image = cv2.VideoCapture('result/small_data/Epoch18/{}'.format(names))
    ret, frame = image.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR)
    frame = np.array(frame)
    frame = Image.fromarray(frame.astype('uint8'),'RGB')
    frame.save('result/small_data/Epoch18_BGR/{}'.format(names))
"""

image = cv2.VideoCapture('result/small_data/Epoch18/126.jpg')
ret, frame = image.read()
#frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR)
frame = np.array(frame)
frame = Image.fromarray(frame.astype('uint8'),'RGB')
frame.save('result/small_data/Epoch18_BGR/126-1.jpg')