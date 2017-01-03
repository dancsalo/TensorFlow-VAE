#!/usr/bin/env

"""
@author: Dan Salo
"""

import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread, imsave
import tqdm


flags = {
    'data_directory': '/media/kd/9ef888dc-a92d-4587-af71-bb562dbc5764/luebeck_2_dataset/',
    'train': ['real_bags', 'tip_weapons_train'],
    'test': ['tip_weapons_test'],
    'views': 4
}


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_features(list_vals):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_vals))


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def process_bags():
    writer = tf.python_io.TFRecordWriter("smiths_bags_train.tfrecords")
    for dataset in flags['train']:
        print("Converting {0}".format(dataset))

        # The summary file containing the IDs of all bags within the dataset
        datafile = flags['data_directory'] + 'description/' + dataset + '.txt'
        with open(datafile) as df:
            # Read in bags one by one from dataset summary file
            for line in df:
                bagID = line.strip()
                # Strip off the .zip if necessary
                if bagID.endswith('.zip'):
                    bagID = bagID[:-4]

                # Parse Images, Labels, and Bounding Box Coordinates
                bagimages = list()
                bbox = list()
                dims = list()
                label = 0
                for view in range(flags['views']):
                    bagimages[view] = convertHiLo2RGB(bagID, view)
                    height, width = bagimages[view].shape
                    dims[view] = [height, width]
                    if dataset == 'tip_weapons_train':
                        left, right, upper, lower = boundWeapon(bagID, view)
                        bbox[view] = [left, right, upper, lower]
                        label = 1
                    else:
                        bbox[view] = [0, 0, 0, 0]
                write_to_tfrecords(label, bbox, bagimages, writer)


def write_to_tfrecords(label, bbox, bagimages, dims, writer):
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': _int64_features(label),
                'view0': _bytes_features(bagimages[0]),
                'dims0': _int64_list_features(dims[0]),
                'bbox0': _int64_list_features(bbox[0]),
                'view1': _bytes_features(bagimages[1]),
                'dims1': _int64_list_features(dims[1]),
                'bbox1': _int64_list_features(bbox[1]),
                'view2': _bytes_features(bagimages[2]),
                'dims2': _int64_list_features(dims[2]),
                'bbox2': _int64_list_features(bbox[2]),
                'view3': _bytes_features(bagimages[3]),
                'dims3': _int64_list_features(dims[3]),
                'bbox3': _int64_list_features(bbox[3]),
            }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)
    

def boundWeapon(bagID, view):
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
    weapon_file = flags['data_directory'] + 'calibrated_pngs/' + bagID + '_' + str(view) + '_gt.png'
    weapon = imread(weapon_file)[:, :, 0]

    # Find the nonzero rows and columns
    rows = np.sum(weapon, axis=0)
    cols = np.sum(weapon, axis=1)

    # Bounding box boundaries
    left = np.nonzero(rows)[0][0]
    right = np.nonzero(rows)[0][-1]
    upper = np.nonzero(cols)[0][0]
    lower = np.nonzero(cols)[0][-1]

    return left, right, upper, lower


def convertHiLo2RGB(bagID, view):
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
    filename_hi = flags['data_directory'] + 'calibrated_pngs/' + bagID + '_' + str(view) + '_hi.png'
    filename_lo = flags['data_directory'] + 'calibrated_pngs/' + bagID + '_' + str(view) + '_lo.png'

    # Read in the hi and lo energy images and perform normalization
    image_hi = transformLogFull(imread(filename_hi))
    image_lo = transformLogFull(imread(filename_lo))

    # Create dummy 3rd channel
    image_mean = (image_hi + image_lo) / 2

    # Stack the 3 channels
    out = np.stack((image_lo, image_hi, image_mean), axis=2)

    return out


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
