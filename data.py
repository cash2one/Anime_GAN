#https://github.com/mrzhu-cool/pix2pix-pytorch/blob/ba031e6040560c2b817f3cedf5eb40e5a9206ccb/data.py

from os.path import join

from dataset import DatasetFromFolder


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir)