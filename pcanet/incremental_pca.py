import torch
from sklearn.utils import gen_batches


class IncrementalPCA:
    def __init__(self, n_components, batch_size=None):
        self.n_components = n_components
        self.batch_size = batch_size
        self.cov = None

    def partial_fit(self, X):
        if self.cov is None:
            self.cov = torch.zeros(X.shape[1], X.shape[1])

        cov_sum = torch.matmul(X, X.transpose(-1, -2)).sum(dim=0)
        cov_sum /= X.shape[0] * X.shape[2]

        self.cov += cov_sum

        return self

    def fit(self, X):
        n_samples = X.shape[0]

        if self.batch_size is None:
            batch_size_ = n_samples
        else:
            batch_size_ = self.batch_size

        for batch in gen_batches(n_samples, batch_size_,
                                 min_batch_size=self.n_components or 0):
            self.partial_fit(X[batch])

        return self

    @property
    def components_(self):
        (e, v) = torch.eig(self.cov, eigenvectors=True)
        _, indicies = torch.sort(e[:, 0], descending=True)
        return v[:, indicies[:self.n_components]]
