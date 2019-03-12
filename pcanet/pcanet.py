import math
import torch
import torch.nn.functional as F
from torch import Tensor


def _extract_image_patches(img: Tensor, filter_size, stride=1, remove_mean=True, dim=0):
    # extract patches
    X = (img.unfold(dim, filter_size, stride)
         .unfold(dim+1, filter_size, stride))

    # unroll patches to vectors
    desired_shape = [-1, filter_size ** 2]
    if dim == 1:
        desired_shape.insert(0, img.shape[0])
    # patch vectors are vertically stacked according to paper
    X = X.reshape(desired_shape).transpose(-1, -2)

    if remove_mean:
        X -= X.mean(dim=dim, keepdim=True)

    return X


class PCANet:
    def __init__(self, num_filters: list, filter_size, hist_blk_size, blk_overlap_ratio):
        self.params = {
            'num_filters': num_filters,
            'filter_size': filter_size,
            'hist_blk_size': hist_blk_size,
            'blk_overlap_ratio': blk_overlap_ratio,
        }
        self.W_1 = None
        self.W_2 = None

    def _PCA_filter_bank(self, X: Tensor, stage):
        # Combining all X_i and performing PCA as described in paper may lead to OOM

        # The following vectorized version may lead to OOM ¯\_(ツ)_/¯
        # cov_sum = torch.matmul(X, X.transpose(-1, -2)).sum(dim=0)
        cov_sum = torch.zeros(X.shape[1], X.shape[1])
        for X_i in X:
            cov_sum += torch.mm(X_i, X_i.t())
        cov_sum /= X.shape[0] * X.shape[2]

        (e, v) = torch.eig(cov_sum, eigenvectors=True)
        _, indicies = torch.sort(e[:, 0], descending=True)
        return v[:, indicies[:self.params['num_filters'][stage - 1]]]

    def _convolution_output(self, imgs: Tensor, filter_bank: Tensor) -> Tensor:
        filter_size = self.params['filter_size']
        # HACK: convert to NCHW
        inputs = imgs[:, None, ...]
        weight = filter_bank.t().reshape(-1, 1, filter_size, filter_size)
        padding = (filter_size - 1) // 2  # same padding
        output = F.conv2d(inputs, weight, padding=padding)
        return output

    def _first_stage(self, imgs: Tensor, train) -> Tensor:
        # grayscale NHW image
        assert imgs.dim() == 3 and imgs.nelement() > 0

        print('PCANet first stage...')

        imgs = imgs.float()
        if train:
            X = _extract_image_patches(imgs, self.params['filter_size'], dim=1)
            self.W_1 = self._PCA_filter_bank(X, 1)
        I = self._convolution_output(imgs, self.W_1)  # I_i^l = I[i, l, ...]
        return I

    def _second_stage(self, I: Tensor, train):
        print('PCANet second stage...')

        # Y = [_extract_image_patches(I[:, l, ...], self.params['filter_size'], dim=1)
        #      for l in range(self.params['num_filters'][0])]
        # Y = torch.cat(Y)
        if train:
            N, L1 = I.shape[:2]
            filter_size = self.params['filter_size']
            Y = torch.empty(L1 * N, filter_size ** 2,
                            (I.shape[2] - filter_size + 1) * (I.shape[3] - filter_size + 1))
            Y_view = Y.view(L1, N, filter_size ** 2,
                            (I.shape[2] - filter_size + 1) * (I.shape[3] - filter_size + 1))
            for l in range(L1):
                I_l = I[:, l, ...]
                Y_view[l] = _extract_image_patches(
                    I_l, filter_size, dim=1)  # N * k1k2 * mn
            self.W_2 = self._PCA_filter_bank(Y, 2)
        O = [self._convolution_output(I_i, self.W_2) for I_i in I]
        return O

    def _output_stage(self, O: list) -> list:
        def heaviside(X: Tensor):
            return (X > 0).to(torch.int32)

        def normal_round(n):
            if n - math.floor(n) < 0.5:
                return math.floor(n)
            return math.ceil(n)

        print('PCANet output stage...')

        N = len(O)
        L1, L2 = O[0].shape[:2]
        map_weights = torch.pow(2, torch.arange(L2, dtype=torch.int32))
        f = []
        for cur, O_i in enumerate(O):  # N
            if (cur + 1) % 100 == 0:
                print('Extracting PCANet feasture of the %dth sample...' %
                      (cur + 1))
            Bhist = []
            for O_i_l in O_i:  # L1
                T_i_l = torch.sum(
                    map_weights[:, None, None] * heaviside(O_i_l), dim=0)  # H*W
                stride = normal_round(
                    (1 - self.params['blk_overlap_ratio']) * self.params['hist_blk_size'])
                blocks = _extract_image_patches(
                    T_i_l, self.params['hist_blk_size'], stride, False).t()
                # torch.histc requires FloatTensor
                blocks = blocks.float()
                B = blocks.shape[0]
                Bhist_i = torch.empty(B * 2 ** L2)
                Bhist_i_view = Bhist_i.view(B, -1)
                for i in range(B):
                    Bhist_i_view[i] = torch.histc(blocks[i], 2 ** L2)
                Bhist.append(Bhist_i)
            Bhist = torch.cat(Bhist)
            f.append(Bhist)
        return torch.stack(f)

    def extract_features(self, imgs: Tensor, train=True):
        I = self._first_stage(imgs, train)
        O = self._second_stage(I, train)
        f = self._output_stage(O)
        return f
