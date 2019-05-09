# %%
import time
import itertools
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from sklearn.utils import gen_batches
from sklearn.svm import LinearSVC
from pcanet import PCANet

# %%
# load data (10000 training, 50000 testing)
n_train = 10000
n_test = 50000
train_set = MNIST('datasets', download=True)
train_set_X = train_set.data[:n_train].float().div(255.0)
train_set_y = train_set.targets[:n_train]
test_set_X = train_set.data[n_train:n_train + n_test].float().div(255.0)
test_set_y = train_set.targets[n_train:n_train + n_test]

# %%
# visualization
nclasses = 10  # number of classes to visualize
nexamples = 10  # number of examples for each class

# Chosing indices from training set images
img_idx = [np.where(train_set_y == class_id)[0][0:nexamples]
           for class_id in range(nclasses)]

# Creating plot with subplots
fig, ax = plt.subplots(nclasses, nexamples, figsize=(15, 15))
fig.set_facecolor('black')
for i, class_id in itertools.product(range(nexamples), range(nclasses)):
    ax[i, class_id].imshow(train_set_X[img_idx[class_id][i]], cmap='gray')
    ax[i, class_id].get_xaxis().set_visible(False)
    ax[i, class_id].get_yaxis().set_visible(False)

plt.savefig('mnist.png', bbox_inches='tight', facecolor='black', pad_inches=0)
plt.show()

# %%
# Convert to NCHW
train_set_X = train_set_X[:, None, ...]
test_set_X = test_set_X[:, None, ...]

# %%
print(' ====== PCANet Training ======= ')
net = PCANet([8, 8], 7, 7, 0.5)
time_start = time.time()
train_features = net.extract_features(train_set_X)
train_features = scipy.sparse.csr_matrix(train_features)
time_end = time.time()
print('Time cost %.2f s' % (time_end - time_start))
del train_set_X

#%%
#visualize PCA filter banks
fig, ax = plt.subplots(2, 8, figsize=(16, 4), facecolor='black')
for stage, i in itertools.product(range(2), range(8)):
    filter_bank = eval('net.W_' + str(stage + 1))[:, i].reshape(7, 7)
    ax[stage, i].imshow(filter_bank, cmap='gray')
    ax[stage, i].get_xaxis().set_visible(False)
    ax[stage, i].get_yaxis().set_visible(False)
plt.subplots_adjust(0, 0, 1, 1)
fig.savefig('mnist_filter_bank.png', facecolor='black', bbox_inches='tight', pad_inches=0)
plt.show()

# %%
print(' ====== Training Linear SVM Classifier ======= ')
time_start = time.time()
svm = LinearSVC()
svm.fit(train_features, train_set_y)
time_end = time.time()
print('Time cost %.2f s' % (time_end - time_start))
del train_features, train_set_y

# %%
print(' ====== PCANet Testing ======= ')
batch_size = 4096
n_correct = 0
for batch in gen_batches(n_test, batch_size):
    test_features = net.extract_features(test_set_X[batch], False)
    test_features = scipy.sparse.csr_matrix(test_features)
    pred_labels = svm.predict(test_features)
    n_correct += np.sum(pred_labels == test_set_y[batch].numpy())
print("Total accuracy = %g%% (%d/%d) (classification)" %
      (n_correct / n_test, n_correct, n_test))
