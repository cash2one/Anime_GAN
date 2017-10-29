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

image = cv2.VideoCapture('dataset/small_data/test/a/119.jpg')
ret, frame = image.read()
#print(frame)
frame_color = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
frame_color = np.array(frame_color)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
frame_gray = np.array(frame_gray)
image1 = Image.fromarray(frame_gray.astype('uint8'), 'L')
image1.save('dataset/small_data/test/a/119.jpg')
image2 = Image.fromarray(frame_color.astype('uint8'), 'RGB')
image2.save('dataset/small_data/test/b/119.jpg')