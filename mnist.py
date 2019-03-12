import numpy as np
from torchvision.datasets import MNIST
from sklearn.svm import SVC
from pcanet import PCANet

# load data
train_set = MNIST('datasets', download=True)
train_set_X = train_set.data[:5000].float().div(255)
train_set_y = train_set.targets[:5000]
test_set_X = train_set.data[5000:5100].float().div(255)
test_set_y = train_set.targets[5000:5100]

print(' ====== PCANet Training ======= ')
net = PCANet([8, 8], 7, 7, 0.5)
train_features = net.extract_features(train_set_X)

print(' ====== Training Linear SVM Classifier ======= ')
svm = SVC(kernel='linear')
svm.fit(train_features, train_set_y)

print(' ====== PCANet Testing ======= ')
test_features = net.extract_features(test_set_X, False)
test_set_pred = svm.predict(test_features)
diff = np.sum(test_set_y.numpy() != test_set_pred)
print('Test error: %f' % (diff / len(test_set_y)))
