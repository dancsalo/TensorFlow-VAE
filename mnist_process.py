# load up some dataset. Could be anything but skdata is convenient.
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
train_data = mnist.train.images
test_data = mnist.test.images
valid_data = mnist.validation.images
train_label = mnist.train.labels
test_label = mnist.test.labels
valid_label = mnist.validation.labels
all_data = [train_data, test_data, valid_data]
all_labels = [train_label, test_label, valid_label]
nums = [55000, 10000, 5000]
all_names = ["train", "test", "valid"]
num_classes = 10


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# one MUST randomly shuffle data before putting it into one of these
# formats. Without this, one cannot make use of tensorflow's great
# out of core shuffling.

for num_labels in [1000]:
    for d in range(3):
        data = all_data[d]
        labels = all_labels[d]
        name = all_names[d]
        num_samples = np.zeros(num_classes)
        writer = tf.python_io.TFRecordWriter("mnist_" + str(num_labels) + "_" + name + ".tfrecords")
        # iterate over each example
        # wrap with tqdm for a progress bar
        for example_idx in tqdm(range(nums[d])):
            pixels = data[example_idx].tostring()
            rows = 28
            cols = 28
            depth = 1
            label = int(labels[example_idx])
            if name == "train":
                true_label = int(labels[example_idx])
                label = 10
                if num_samples[true_label] < 10:
                    label = true_label
                    num_samples[label] += 1
                    print(num_samples)
                    print(example_idx)
            # construct the Example proto boject
            example = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                  # Features contains a map of string to Feature proto objects
                  feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': _int64_features(label),
                    'height': _int64_features(rows),
                    'width': _int64_features(cols),
                    'depth': _int64_features(depth),
                    'image': _bytes_features(pixels)
            }))
            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            writer.write(serialized)
