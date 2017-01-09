#!/usr/bin/env

"""
@author: Dan Salo
"""

import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread, imsave, imresize, imrotate
from tqdm import tqdm
import random


flags = {
    'unlabeled_data_dir': '/media/kd/9ef888dc-a92d-4587-af71-bb562dbc5764/real/',
    'labeled_data_dir': '/media/kd/9ef888dc-a92d-4587-af71-bb562dbc5764/luebeck_2_dataset/',
    'save_dir': '/media/kd/6fbd8619-1829-4831-9946-a10df9cc265c/',
    'train': ['tip_weapons_train'],
    'test': ['tip_weapons_test', 'real_bags'],
    'views': 4
}


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_features(list_vals):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_vals))


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
    process_train_labeled_bags()

def process_test_bags():
    writer = tf.python_io.TFRecordWriter(flags['save_dir'] + "smiths_bags_test.tfrecords")
    examples = list()

    for dataset in flags['test']:
        examples_dataset = list()
        print("Converting {0}".format(dataset))

        # The summary file containing the IDs of all bags within the dataset
        datafile = flags['labeled_data_dir'] + 'description/' + dataset + '.txt'
        directory = flags['labeled_data_dir'] + 'calibrated_pngs/'
        with open(datafile) as df:
            # Read in bags one by one from dataset summary file
            for line in df:
                bagID = line.strip()
                # Strip off the .zip if necessary
                if bagID.endswith('.zip'):
                    bagID = bagID[:-4]

                # Parse Images, Labels, and Bounding Box Coordinates
                examples_dataset.append((directory + bagID, dataset))
        print("%d Records in %s dataset" % (len(examples_dataset), dataset))
        examples.extend(examples_dataset)
    print('Writing %d Cases to TFRecords file.' % len(examples))
    for example_idx in tqdm(range(len(examples))):
        write_bag(examples[example_idx][0], examples[example_idx][1], writer)

def process_train_labeled_bags():
    writer = tf.python_io.TFRecordWriter(flags['save_dir'] + "smiths_bags_train_labeled.tfrecords")
    examples = list()

    # pull from lubeck dataset
    for dataset in flags['train']:
        print("Converting {0}".format(dataset))

        # The summary file containing the IDs of all bags within the dataset
        datafile = flags['labeled_data_dir'] + 'description/' + dataset + '.txt'
        directory = flags['labeled_data_dir'] + 'calibrated_pngs/'
        with open(datafile) as df:
            # Read in bags one by one from dataset summary file
            for line in df:
                bagID = line.strip()
                # Strip off the .zip if necessary
                if bagID.endswith('.zip'):
                    bagID = bagID[:-4]

                # Parse Images, Labels, and Bounding Box Coordinates
                examples.append((directory + bagID, dataset))
    random.shuffle(examples)
    print('Writing %d Cases to TFRecords file.' % len(examples))
    for example_idx in tqdm(range(100)):
        write_bag(examples[example_idx][0], examples[example_idx][1], writer)

def process_train_unlabeled_bags():
    writer = tf.python_io.TFRecordWriter(flags['save_dir'] + "smiths_bags_train_unlabeled.tfrecords")
    examples = list()

    # pull from real bag dataset
    years = os.listdir(flags['unlabeled_data_dir'])
    dataset = 'real_bags'
    for y in years:
        print('Now processing %s ..' % y)
        index1 = os.listdir(flags['unlabeled_data_dir'] + y)
        for i in index1:
            index2 = os.listdir(flags['unlabeled_data_dir'] + y + '/' + i)
            for ii in index2:
                png_directory = flags['unlabeled_data_dir'] + y + '/' + i + '/' + ii + '/'
                pngs = os.listdir(png_directory)

                # Loop through and make list of bagids
                bagids = list()
                for p in pngs:
                    splitf = p.split('_')
                    if splitf[4] == '0' and splitf[5] == 'hi.png':
                        bagID = splitf[0] + '_' +splitf[1] + '_' +splitf[2] + '_' + splitf[3]                           
                        bagids.append(bagID)
                        print(bagID)
                # Write each record to TFrecords file
                for bagID in bagids:
                    examples.append((png_directory + bagID, dataset))
    random.shuffle(examples)
    for example_idx in tqdm(range(len(examples))):
        write_bag(examples[example_idx][0], examples[example_idx][1], writer)

def write_bag(filepath, dataset, writer):
    bagimages = list()
    bbox = list()
    dims = list()
    label = 0
    for view in range(flags['views']):
        img, _  = convertHiLo2RGB(filepath, view)
        bagimages.append(img)
        dims.append([img.shape[0], img.shape[1]])
        if dataset == 'tip_weapons_test' or dataset == 'tip_weapons_train':
            left, right, upper, lower = boundWeapon(filepath, view)
            bbox.append([left, right, upper, lower])
            label = 1
        else:
            bbox.append([0, 0, 0, 0])
    write_to_tfrecords(label, bbox, bagimages, dims, writer)


def write_to_tfrecords(label, bbox, bagimages, dims, writer):
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': _int64_features(label),
                'view0': _bytes_features(bagimages[0].tobytes()),
                'dims0': _int64_list_features(dims[0]),
                'bbox0': _int64_list_features(bbox[0]),
                'view1': _bytes_features(bagimages[1].tobytes()),
                'dims1': _int64_list_features(dims[1]),
                'bbox1': _int64_list_features(bbox[1]),
                'view2': _bytes_features(bagimages[2].tobytes()),
                'dims2': _int64_list_features(dims[2]),
                'bbox2': _int64_list_features(bbox[2]),
                'view3': _bytes_features(bagimages[3].tobytes()),
                'dims3': _int64_list_features(dims[3]),
                'bbox3': _int64_list_features(bbox[3]),
            }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)
    

def boundWeapon(filepath, view):
    '''
    Find the bounding box boundaries of a weapon mask

    Args:
        bagID (string): Identifier for the bag found in .txt file in descriptions
        view (int):     Which of the four view of the scanner

    Returns:
        left (int): left border coordinate
        right (int): right border coordinate
        upper (int): upper border coordinate
        lower (int): lower border coordinate
    '''
    weapon_file = filepath + '_' + str(view) + '_gt.png'
    weapon = imread(weapon_file)[:, :, 0]

    # Find the nonzero rows and columns
    rows = np.sum(weapon, axis=0)
    cols = np.sum(weapon, axis=1)

    # Bounding box boundaries
    left = np.nonzero(rows)[0][0]
    right = np.nonzero(rows)[0][-1]
    upper = np.nonzero(cols)[0][0]
    lower = np.nonzero(cols)[0][-1]

    return left.item(), right.item(), upper.item(), lower.item()


def convertHiLo2RGB(filepath, view):
    '''
    Bag images are 2 channels (hi,lo). To use weights pre-trained on ImageNet,
    inputs must be 3 channels (RGB). This function combines the hi-lo channels
    and artificially creates a third by taking the mean of the two. Also
    performs image normalization using transformLogFull


    Note: not used for weapon masks

    Args:
        bagID (string): Identifier for the bag found in .txt file in descriptions
        view (int):     Which of the four view of the scanner

    Returns:
        out (numpy array 3D): 3 channel image (hi,lo,mean)
    '''

    filename_hi = filepath + '_' + str(view) + '_hi.png'
    filename_lo = filepath + '_' + str(view) + '_lo.png'

    # Read in the hi and lo energy images and perform normalization
    image_hi = transformLogFull(imread(filename_hi))
    image_lo = transformLogFull(imread(filename_lo))
    if image_hi.shape[0] > image_hi.shape[1]:
        size = image_hi.shape[1]
        dim = 1
    else:
        size = image_hi.shape[0]
        dim = 2
    image_hi_resize = imresize(image_hi, 256.1/size)
    image_lo_resize = imresize(image_lo, 256.1/size)

    # Create dummy 3rd channel
    image_mean = (image_hi_resize + image_lo_resize) / 2

    # Stack the 3 channels
    out = np.stack((image_lo_resize, image_hi_resize, image_mean), axis=2)

    return out, dim


def transformLogFull(image):
    '''
    Bag images at low and high energies is 16 bits; normalize these using a log

    Note: not necessary for weapon masks

    Args:
        image (numpy array 2D): image to be normalized

    Returns:
        Normalized image
    '''
    maxVal = 65536
    maxValInv = 1. / maxVal
    scaling = 65535. / np.log(maxValInv)

    return np.minimum(maxVal, np.log((image + 1) * maxValInv) * scaling)

if __name__ == '__main__':
    main()

