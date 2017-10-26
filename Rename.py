from os import listdir, rename
from util import is_image_file

a_filenames = [x for x in listdir('dataset/dataset/test/a') if is_image_file(x)]
i=9985
for x in a_filenames:
    i += 1
    rename('dataset/dataset/test/a/{}'.format(x),'dataset/dataset/test/a/{}.png'.format(i))
"""
i=0
for x in b_filenames:
    i += 1
    rename('dataset/dataset/train/b/{}'.format(x), 'dataset/dataset/train/b/{}.png'.format(i))
"""