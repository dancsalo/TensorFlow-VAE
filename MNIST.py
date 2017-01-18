#!/usr/bin/env python

"""
@author: Dan Salo, Jan 2017

Purpose: Create partially-labeled MNIST dataset in .tfrecords format. Number of labels specified by user.

Modules:
    convert_data_tfrecords()
    aux_convert_tfreocrds()
    write()
"""

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import argparse
import os
import random


# Global Flag Dictionary
def main():
    """Downloads and Converts MNIST dataset to four .tfrecords files (train_labeled, train_unlabeled, valid, test)
    Takes variable number of labels."""

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-l', '--label_list', nargs='+', type=int, default=[1000])
    parser.add_argument('-d', '--dir', default='data/')
    args = vars(parser.parse_args())

    # Load and Convert Data
    all_data, all_labels = load_data()
    make_directory(args['dir'])
    convert_data_tfrecords(all_data, all_labels, args['label_list'], args['dir'])


flags = {
    'data_directory': 'data/',
    'nums': [55000, 10000, 5000],
    'all_names': ["train", "test", "valid"],
    'num_classes': 10
}


def convert_data_tfrecords(all_data, all_labels, list_num_labels, data_directory):
    """ Saves MNIST images and labels in .tfrecords format with two train files (labeled, unlabeled)
    :param all_data: list of train, test, and validation pre-loaded images
    :param all_labels: list of train, test, and validation pre-loaded labels
    :param list_num_labels: list of number of labels for each generated dataset (i.e. 100, 1000)
    :param data_directory: string of where .tfrecords files will be saved
    """

    # Loop through [train, valid, test] for all number of labeled images
    for num_labels in list_num_labels:
        for d in range(len(all_data)):

            # Initialize
            data = all_data[d]
            labels = all_labels[d]
            name = flags['all_names'][d]
            num_samples = np.zeros(flags['num_classes'])
            examples_labeled = list()
            examples_unlabeled = list()
            examples = list()

            # Create writers
            if name == 'train':
                writer_labeled = tf.python_io.TFRecordWriter(data_directory + 'mnist_' + str(num_labels) + "_" +
                                                             name + "_labeled.tfrecords")
                writer_unlabeled = tf.python_io.TFRecordWriter(data_directory + 'mnist_' + str(num_labels) + "_" +
                                                               name + "_unlabeled.tfrecords")
            else:
                writer = tf.python_io.TFRecordWriter(flags['data_directory'] + 'mnist_' + str(num_labels) + "_" +
                                                     name + ".tfrecords")

            # Iterate over each example
            for example_idx in range(flags['nums'][d]):
                pixels = data[example_idx].tostring()
                label_np = labels[example_idx].astype("int32")
                label = label_np.tolist()

                # Write example to file via writer object
                if name == "train": 
                    if num_samples[label_np == 1] < num_labels:
                        num_samples[label_np == 1] += 1
                        examples_labeled.append((pixels, label))
                    else:
                        label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        examples_unlabeled.append((pixels, label))
                else:
                    examples.append((pixels, label))

            # Shuffle all examples. This is imperative for good mixing with TF queueing and shuffling.
            if name == "train":
                random.shuffle(examples_labeled)
                random.shuffle(examples_unlabeled)
            else:
                random.shuffle(examples)

            # Iterate over all examples and save each to .tfrecords file
            if name == "train":
                for idx_labeled in tqdm(range(len(examples_labeled))):
                    write(examples_labeled[idx_labeled][0], examples_labeled[idx_labeled][1], writer_labeled)
                for idx_unlabeled in tqdm(range(len(examples_unlabeled))):
                    write(examples_unlabeled[idx_unlabeled][0], examples_unlabeled[idx_unlabeled][1], writer_unlabeled)
            else:
                for idx in tqdm(range(len(examples))):
                    write(examples[idx][0], examples[idx][1], writer)


def aux_convert_tfrecords(all_data, all_labels, list_num_labels, data_directory):
    """ Saves MNIST images and labels in .tfrecords format with one train file
    This function is not used in most of our semi-supervised models as we want to have balance labels in minibatches

    :param all_data: list of train, test, and validation pre-loaded images
    :param all_labels: list of train, test, and validation pre-loaded labels
    :param list_num_labels: list of number of labels for each generated dataset (i.e. 100, 1000)
    :param data_directory: string of where .tfrecords files will be saved
    """

    # Loop through [train, valid, test] for all number of labeled images
    for num_labels in list_num_labels:
        for d in range(len(all_data)):

            # Initialize
            data = all_data[d]
            labels = all_labels[d]
            name = flags['all_names'][d]
            num_samples = np.zeros(flags['num_classes'])
            examples = list()

            # Create writer object
            writer = tf.python_io.TFRecordWriter(data_directory + "mnist_" + str(num_labels) + "_" +
                                                 name + ".tfrecords")

            # Iterate over each example and append to list
            for example_idx in range(flags['nums'][d]):

                pixels = data[example_idx].tostring()
                label_np = labels[example_idx].astype("int32")
                label = label_np.tolist()

                if name == "train": 
                    if num_samples[label_np == 1] < num_labels:
                        num_samples[label_np == 1] += 1
                    else:
                        label = [0,0,0,0,0,0,0,0,0,0]
                examples.append((pixels, label))

            # Shuffle all examples. This is imperative for good mixing with TF queueing and shuffling.
            random.shuffle(examples)

            # Iterate over all examples and save each to .tfrecords file
            for idx in tqdm(range(len(examples))):
                write(examples[idx][0], examples[idx][1], writer)


def write(pixels, label, writer):
    """Write image pixels and label from one example to .tfrecords file"""
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'label': _int64_list_features(label),
            'height': _int64_features(28),
            'width': _int64_features(28),
            'depth': _int64_features(1),
            'image': _bytes_features(pixels)
    }))
    # Use the proto object to serialize the example to a string and write to disk
    serialized = example.SerializeToString()
    writer.write(serialized)


def load_data():
    """ Download MNIST data from TensorFlow package """
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train_data = mnist.train.images
    test_data = mnist.test.images
    valid_data = mnist.validation.images
    train_label = mnist.train.labels
    test_label = mnist.test.labels
    valid_label = mnist.validation.labels
    all_data = [train_data, test_data, valid_data]
    all_labels = [train_label, test_label, valid_label]
    return all_data, all_labels


def make_directory(folder_path):
    """Creates directory if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def _int64_features(value):
    """Value takes a the form of a single integer"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_features(list_ints):
    """Value takes a the form of a list of integers"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_ints))


def _bytes_features(value):
    """Value takes the form of a string of bytes data"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    main()
