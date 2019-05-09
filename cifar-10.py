#%%
import time
import itertools
import numpy as np
import scipy.sparse
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from sklearn.utils import gen_batches
from sklearn.svm import LinearSVC
from pcanet import PCANet

#%%
# load data
n_train = 1000
n_test = 1000
train_set = CIFAR10('datasets', download=True)
train_set_X = torch.from_numpy(train_set.data[::50]).float().div(255)
train_set_y = np.asarray(train_set.targets[::50])

test_set = CIFAR10('datasets', False, download=True)
test_set_X = torch.from_numpy(test_set.data[::10]).float().div(255.0)
test_set_y = np.asarray(test_set.targets[::10])

# %%
# visualization
nclasses = 10  # number of classes to visualize
nexamples = 10  # number of examples for each class

# Chosing indices from training set images
img_idx = [np.where(train_set_y == class_id)[0][0:nexamples]
           for class_id in range(nclasses)]

# Creating plot with subplots
fig, ax = plt.subplots(nclasses, nexamples, figsize=(15, 15))

for axis, label in zip(ax[0], train_set.classes):
    axis.set_title(label)

for i, class_id in itertools.product(range(nexamples), range(nclasses)):
    ax[i, class_id].imshow(train_set_X[img_idx[class_id][i]])
    ax[i, class_id].get_xaxis().set_visible(False)
    ax[i, class_id].get_yaxis().set_visible(False)

plt.savefig('cifar10.png', bbox_inches='tight')
plt.show()

#%%
# Convert to NCHW
train_set_X = train_set_X.permute(0, 3, 1, 2)
test_set_X = test_set_X.permute(0, 3, 1, 2)

#%%
print(' ====== PCANet Training ======= ')
# net = PCANet([40, 8], 5, 8, 0.5)
net = PCANet([40, 8], 5, 8, 0.5, [4, 2, 1])
time_start = time.time()
train_features = net.extract_features(train_set_X)
train_features = scipy.sparse.csr_matrix(train_features)
time_end = time.time()
print('Time cost %.2f s' % (time_end - time_start))
del train_set_X

#%%
#visualize PCA filter banks
fig, ax = plt.subplots(5, 8, figsize=(16, 10), facecolor='black')
for r, c in itertools.product(range(5), range(8)):
    # convert to HWC
    filter_bank = net.W_1[:, r * 8 + c].reshape(3, 5, 5).permute(1, 2, 0)
    #scale to [0, 1]
    filter_bank = np.interp(filter_bank, (filter_bank.min(), filter_bank.max()), (0, 1))
    ax[r, c].imshow(filter_bank)
    ax[r, c].get_xaxis().set_visible(False)
    ax[r, c].get_yaxis().set_visible(False)
plt.subplots_adjust(0, 0, 1, 1)
fig.savefig('cifar10_w1.png', facecolor='black', bbox_inches='tight', pad_inches=0)
plt.show()

fig, ax = plt.subplots(1, 8, figsize=(16, 2), facecolor='black')
for i in range(8):
    filter_bank = net.W_2[:, i].reshape(5, 5)
    ax[i].imshow(filter_bank, cmap='gray')
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
plt.subplots_adjust(0, 0, 1, 1)
fig.savefig('cifar10_w2.png', facecolor='black', bbox_inches='tight', pad_inches=0)
plt.show()

#%%
print(' ====== Training Linear SVM Classifier ======= ')
time_start = time.time()
svm = LinearSVC(C=10)
svm.fit(train_features, train_set_y)
time_end = time.time()
print('Time cost %.2f s' % (time_end - time_start))
del train_features, train_set_y

#%%
print(' ====== PCANet Testing ======= ')
batch_size = 4096
n_correct = 0
for batch in gen_batches(n_test, batch_size):
    test_features = net.extract_features(test_set_X[batch], False)
    test_features = scipy.sparse.csr_matrix(test_features)
    pred_labels = svm.predict(test_features)
    n_correct += np.sum(np.asarray(pred_labels) == test_set_y[batch])
print("Total accuracy = %g%% (%d/%d) (classification)" %
      (n_correct / n_test, n_correct, n_test))
