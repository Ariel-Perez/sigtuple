"""Main script."""
from easyio.filepath import get_files
from PIL import Image
import numpy as np
import tensorflow as tf


def read_files(training_path):
    """Read the training files, return numpy data."""
    training_samples = [
        x for x in get_files(training_path)
        if '-mask' not in x and 'train' in x
    ]

    data = []
    targets = []

    for sample_path in training_samples:
        print(sample_path)
        target_path = sample_path.replace('.jpg', '-mask.jpg')
        sample = Image.open(sample_path)
        target = Image.open(target_path)

        h, w = sample.size
        pix = sample.load()
        pixels = [
            [list(pix[y, x]) for x in range(w)]
            for y in range(h)
        ]

        pix_target = target.load()
        pixels_target = [
            [pix_target[y, x] for x in range(w)]
            for y in range(h)
        ]

        data.append(pixels)
        targets.append(pixels_target)

    x = np.array(data, np.int32)
    y = np.array(targets, np.int32)
    return x, y


def save_data(training_path, data, masks):
    """Read the data and save it."""
    x, y = read_files(training_path)
    np.save(data, x)
    np.save(masks, y)


if __name__ == '__main__':
    # train_path = 'C:/Users/Admin/Downloads/SigTuple/SigTuple_data/Train_Data'
    # save_data(train_path, 'data', 'masks')
    x = np.load('data.npy')
    y = np.load('masks.npy')
