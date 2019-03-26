import numpy as np
import scipy.sparse
import torch
from torchvision.datasets import CIFAR10
from sklearn.utils import gen_batches
from liblinear import liblinearutil
from pcanet import PCANet

# load data
n_train = 1000
n_test = 1000
train_set = CIFAR10('datasets', download=True)
train_set_X = torch.from_numpy(train_set.data[:n_train]).float().div(255)
train_set_y = train_set.targets[:n_train]

test_set = CIFAR10('datasets', False, download=True)
test_set_X = torch.from_numpy(test_set.data[:n_test]).float().div(255.0)
test_set_y = test_set.targets[:n_test]

# Convert to NCHW
train_set_X = train_set_X.permute(0, 3, 1, 2)
test_set_X = test_set_X.permute(0, 3, 1, 2)

print(' ====== PCANet Training ======= ')
net = PCANet([40, 8], 5, 8, 0.5)
# net = PCANet([40, 8], 5, 8, 0.5, [4, 2, 1])
train_features = net.extract_features(train_set_X)
train_features = scipy.sparse.csr_matrix(train_features)
del train_set_X

print(' ====== Training Linear SVM Classifier ======= ')
model = liblinearutil.train(
    train_set_y, train_features, '-s 1 -c 10 -q')
del train_features, train_set_y

print(' ====== PCANet Testing ======= ')
batch_size = 4096
n_correct = 0
for batch in gen_batches(n_test, batch_size):
    test_features = net.extract_features(test_set_X[batch], False)
    test_features = scipy.sparse.csr_matrix(test_features)
    pred_labels, (acc, mse, scc), pred_values = liblinearutil.predict(
        test_set_y[batch], test_features, model)
    n_correct += np.sum(np.asarray(pred_labels) == test_set_y[batch])
print("Total accuracy = %g%% (%d/%d) (classification)" %
      (n_correct / n_test, n_correct, n_test))
