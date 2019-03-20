import math
import torch
import torch.nn.functional as F
from torch import Tensor
from incremental_pca import IncrementalPCA


class PCANet:
    def __init__(self, num_filters: list, filter_size, hist_blk_size, blk_overlap_ratio, batch_size=256):
        self.params = {
            'num_filters': num_filters,
            'filter_size': filter_size,
            'hist_blk_size': hist_blk_size,
            'blk_overlap_ratio': blk_overlap_ratio,
        }
        self.W_1 = None
        self.W_2 = None
        self.batch_size = batch_size

    def _convolution_output(self, imgs: Tensor, filter_bank: Tensor) -> Tensor:
        filter_size = self.params['filter_size']
        # HACK: convert to NCHW
        inputs = imgs[:, None, ...]
        weight = filter_bank.t().reshape(-1, 1, filter_size, filter_size)
        padding = (filter_size - 1) // 2  # same padding
        output = F.conv2d(inputs, weight, padding=padding)
        return output

    @staticmethod
    def _extract_image_patches(img: Tensor, filter_size, stride=1, remove_mean=True, dim=0):
        # extract patches
        X = (img.unfold(dim, filter_size, stride)
             .unfold(dim + 1, filter_size, stride))

        # unroll patches to vectors
        desired_shape = [-1, filter_size ** 2]
        if dim == 1:
            desired_shape.insert(0, img.shape[0])
        # patch vectors are vertically stacked according to paper
        X = X.reshape(desired_shape).transpose(-1, -2)

        if remove_mean:
            X -= X.mean(dim=dim, keepdim=True)

        return X

    @staticmethod
    def conv_output_size(w, filter_size, padding=0, stride=1):
        return int((w - filter_size + 2 * padding) / stride + 1)

    def _first_stage(self, imgs: Tensor, train) -> Tensor:
        # grayscale NHW image
        assert imgs.dim() == 3 and imgs.nelement() > 0

        print('PCANet first stage...')

        if train:
            X = self._extract_image_patches(
                imgs, self.params['filter_size'], dim=1)
            # self.W_1 = self._PCA_filter_bank(X, 1)
            self.W_1 = IncrementalPCA(
                self.params['num_filters'][0]).fit(X).components_
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
                            self.conv_output_size(I.shape[2], filter_size) *
                            self.conv_output_size(I.shape[3], filter_size))
            Y_view = Y.view(L1, N, filter_size ** 2,
                            self.conv_output_size(I.shape[2], filter_size) *
                            self.conv_output_size(I.shape[3], filter_size))
            for l in range(L1):
                I_l = I[:, l, ...]
                Y_view[l] = self._extract_image_patches(
                    I_l, filter_size, dim=1)  # N * k1k2 * mn
            # self.W_2 = self._PCA_filter_bank(Y, 2)
            self.W_2 = IncrementalPCA(
                self.params['num_filters'][1], self.batch_size).fit(Y).components_
        O = [self._convolution_output(I_i, self.W_2) for I_i in I]
        return O

    def _output_stage(self, O: list) -> Tensor:
        def heaviside(X: Tensor):
            return (X > 0).to(torch.int32)

        def normal_round(n):
            if n - math.floor(n) < 0.5:
                return math.floor(n)
            return math.ceil(n)

        print('PCANet output stage...')

        N = len(O)
        L1, L2 = O[0].shape[:2]

        map_weights = torch.pow(2, torch.arange(
            L2, dtype=torch.int32)).flip(dims=(0,))
        stride = normal_round(
            (1 - self.params['blk_overlap_ratio']) * self.params['hist_blk_size'])
        H, W = O[0].shape[-2:]
        B = (self.conv_output_size(H, self.params['hist_blk_size'], stride=stride) *
             self.conv_output_size(W, self.params['hist_blk_size'], stride=stride))
        f = torch.empty(N, L1, B, 2 ** L2)
        for i, O_i in enumerate(O):  # N
            if (i + 1) % 100 == 0:
                print('Extracting PCANet feature of the %dth sample...' % (i + 1))
            Bhist_i = f[i]
            for l, O_i_l in enumerate(O_i):  # L1
                T_i_l = torch.sum(
                    map_weights[:, None, None] * heaviside(O_i_l), dim=0)  # H*W
                blocks = self._extract_image_patches(
                    T_i_l, self.params['hist_blk_size'], stride, False).t()
                # torch.histc requires FloatTensor
                blocks = blocks.float()

                for b in range(B):
                    Bhist_i[l, b] = torch.histc(
                        blocks[b], 2 ** L2, min=0, max=255)
        return f.flatten(start_dim=1)

    def extract_features(self, imgs: Tensor, train=True):
        I = self._first_stage(imgs, train)
        O = self._second_stage(I, train)
        f = self._output_stage(O)
        return f
