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
# train_images = train_images / 255.0
# test_images = test_images / 255.0

train_images = train_images[:1000]
train_labels = train_labels[:1000] 
test_images = test_images[:1000]
test_labels = test_labels[:1000] 

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

def train(image, label, learning_rate=0.005):
    out, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1/out[label]

    gradient = softmax.backward(gradient, learning_rate)
    gradient = pool.backward(gradient)
    gradient = conv.backward(gradient, learning_rate)

    return loss, acc

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i > 0 and i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)