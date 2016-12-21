# load up some dataset. Could be anything but skdata is convenient.
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
train_data = mnist.train.next_batch(mnist.train.num_examples)
test_data = mnist.test.next_batch(mnist.test.num_examples)
valid_data = mnist.validation.next_batch(mnist.validation.num_examples)
all_data = (train_data, test_data, valid_data)
all_names = ["train", "test", "valid"]
num_classes = 10


# one MUST randomly shuffle data before putting it into one of these
# formats. Without this, one cannot make use of tensorflow's great
# out of core shuffling.

for num_labels in [100, 300, 1000, 5000]:
    for d in range(3):
        data = all_data[d]
        name = all_names[d]
        num_samples = np.zeros(num_classes)
        writer = tf.python_io.TFRecordWriter("mnist_" + str(num_labels) + "_" + name + ".tfrecords")
        # iterate over each example
        # wrap with tqdm for a progress bar
        for example_idx in tqdm(range(mnist.train.num_examples)):
            pixels = data[0][example_idx].astype("int64")
            true_label = data[1][example_idx].astype("int64")
            label = np.zeros(num_classes).astype("int64")

            if num_samples[true_label == 1] < num_classes:
                label = true_label
                num_samples[label == 1] += 1
            # construct the Example proto boject
            example = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                  # Features contains a map of string to Feature proto objects
                  feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=label.tolist())),
                    'image': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=pixels.tolist())),
            }))
            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            writer.write(serialized)
