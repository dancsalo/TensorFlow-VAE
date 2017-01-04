# load up some dataset. Could be anything but skdata is convenient.
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import random

flags = {
    'labels_list': [100, 300, 500, 1000, 3000, 5000],
    'data_directory': 'data/',
    'nums': [55000, 10000, 5000],
    'all_names': ["train", "test", "valid"],
    'num_classes': 10
}

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_features(list_vals):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_vals))

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
    all_data, all_labels = load_data()
    make_directory(flags['data_directory'])
    semi_all_train(flags['labels_list'], all_data, all_labels)
    # semi_split_train(flags['labels_list'], all_data, all_labels)

def semi_split_train(labels_list, all_data, all_labels):
    for num_labels in labels_list:
        for d in range(3):
            data = all_data[d]
            labels = all_labels[d]
            name = flags['all_names'][d]
            num_samples = np.zeros(flags['num_classes'])
            if name == "train":
                writer_labeled = tf.python_io.TFRecordWriter(flags['data_directory'] + "mnist_" + str(num_labels) + "_" + name + "_labeled.tfrecords")
                writer_unlabeled = tf.python_io.TFRecordWriter(flags['data_directory'] + "mnist_" + str(num_labels) + "_" + name + "_unlabeled.tfrecords")
            else:
                writer = tf.python_io.TFRecordWriter(flags['data_directory'] + "mnist_" + str(num_labels) + "_" + name + ".tfrecords")
            # iterate over each example
            # wrap with tqdm for a progress bar
            for example_idx in tqdm(range(flags['nums'][d])):
                pixels = data[example_idx].tostring()
                label_np = labels[example_idx].astype("int32")
                label = label_np.tolist()
                if name == "train": 
                    if num_samples[label_np == 1] < num_labels:
                        num_samples[label_np == 1] += 1
                        write(pixels, label, writer_labeled)
                    else:
                        label = [0,0,0,0,0,0,0,0,0,0]
                        write(pixels, label, writer_unlabeled)
                else:
                    write(pixels, label, writer)

def semi_all_train(labels_list, all_data, all_labels):
    random.seed(10)
    for num_labels in labels_list:
        for d in range(3):
            data = all_data[d]
            labels = all_labels[d]
            name = flags['all_names'][d]
            num_samples = np.zeros(flags['num_classes'])
            writer = tf.python_io.TFRecordWriter(flags['data_directory'] + "mnist_" + str(num_labels) + "_" + name + ".tfrecords")
            # iterate over each example
            # wrap with tqdm for a progress bar
            examples = list()
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
            random.shuffle(examples)
            for idx in tqdm(range(flags['nums'][d])):
                write(examples[idx][0], examples[idx][1], writer)


def write(pixels, label, writer):
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
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)

def load_data():
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
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == "__main__":
    main()
