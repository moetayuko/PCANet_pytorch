import numpy as np
from torchvision.datasets import MNIST
from liblinear import liblinearutil
from pcanet import PCANet

# load data (10000 training, 50000 testing)
n_train = 10000
n_test = 50000
train_set = MNIST('datasets', download=True)
train_set_X = train_set.data[:n_train].float().div(255.0)
train_set_y = train_set.targets[:n_train]
test_set_X = train_set.data[n_train:n_train + n_test].float().div(255.0)
test_set_y = train_set.targets[n_train:n_train + n_test]

print(' ====== PCANet Training ======= ')
net = PCANet([8, 8], 7, 7, 0.5)
train_features = net.extract_features(train_set_X)

print(' ====== Training Linear SVM Classifier ======= ')
model = liblinearutil.train(
    train_set_y.numpy(), train_features.numpy(), '-s 1 -q')

print(' ====== PCANet Testing ======= ')
test_features = net.extract_features(test_set_X, False)
pred_labels, (acc, mse, scc), pred_values = liblinearutil.predict(
    test_set_y.numpy(), test_features.numpy(), model)
