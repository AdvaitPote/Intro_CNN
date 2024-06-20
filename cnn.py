import mnist
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import tensorflow as tf
import numpy as np

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

# Split the data into training and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images[:1000]
train_labels = train_labels[:1000] 

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13*13*8, 10)

def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    accuracy = 1 if np.argmax(out) == label else 0

    return out, loss, accuracy

loss = 0
num_correct = 0
for i, (image, label) in enumerate(zip(test_images, test_labels)):
    _, l, accuracy = forward(image, label)
    loss += l
    num_correct += accuracy
    if i % 100 == 99:
        print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

