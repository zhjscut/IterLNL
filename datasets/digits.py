import numpy as np
from scipy.io import loadmat
import sys
import gzip
import _pickle as cPickle

def load_mnist(dataroot, scale=True, usps=False, all_use=False):
    # Process MNIST
    mnist_data = loadmat('{}/mnist_data.mat'.format(dataroot))
    if scale:
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test =  mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    # inds = np.random.permutation(mnist_train.shape[0])
    # mnist_train = mnist_train[inds]
    # train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    if usps and not all_use:
        # make a class-balanced subset
        select_idx = []
        for i in range(10):
            select_idx += list(np.where(train_label == i)[0][:200])
        mnist_train = mnist_train[select_idx]
        train_label = train_label[select_idx]

    mnist_train = mnist_train.astype(np.float64) / 255
    mnist_test = mnist_test.astype(np.float64) / 255
    return mnist_train, train_label, mnist_test, test_label


def load_svhn(dataroot):
    svhn_train = loadmat('{}/train_32x32.mat'.format(dataroot))
    svhn_test = loadmat('{}/test_32x32.mat'.format(dataroot))

    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = svhn_train['y'].squeeze(1)
    # label of raw data range from [1, 10], which need to be correct to [0, 9]
    svhn_label[svhn_label == 10] = 0

    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = svhn_test['y'].squeeze(1)
    svhn_label_test[svhn_label_test == 10] = 0

    svhn_train_im = svhn_train_im.astype(np.float64) / 255
    svhn_test_im = svhn_test_im.astype(np.float64) / 255
    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test


def load_usps(dataroot, all_use=False):
    f = gzip.open('{}/usps_28x28.pkl'.format(dataroot), 'rb')
    data_set = cPickle.load(f, encoding='latin1')
    f.close()
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]

    # inds = np.random.permutation(img_train.shape[0])
    # if all_use:
    #     img_train = img_train[inds][:6562]
    #     label_train = label_train[inds][:6562]
    # else:
    #     img_train = img_train[inds][:1800]
    #     label_train = label_train[inds][:1800]

    if all_use:
        img_train = img_train[:6562]
        label_train = label_train[:6562]
    else:
        # make a class-balanced subset
        select_idx = []
        for i in range(10):
            select_idx += list(np.where(label_train == i)[0][:180])
        img_train = img_train[select_idx]
        label_train = label_train[select_idx]

    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))

    img_train = img_train.astype(np.float64) / 255
    img_test = img_test.astype(np.float64) / 255
    return img_train, label_train, img_test, label_test
