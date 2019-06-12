import math
import torch
import torch.nn.functional as F
from torch import Tensor
from .incremental_pca import IncrementalPCA


class PCANet:
    def __init__(self, num_filters: list, filter_size, hist_blk_size, blk_overlap_ratio, pyramid=None, batch_size=256):
        self.params = {
            'num_filters': num_filters,
            'filter_size': filter_size,
            'hist_blk_size': hist_blk_size,
            'blk_overlap_ratio': blk_overlap_ratio,
            'pyramid': pyramid,
        }
        self.W_1 = None
        self.W_2 = None
        self.batch_size = batch_size

    def _convolution_output(self, imgs: Tensor, filter_bank: Tensor) -> Tensor:
        filter_size = self.params['filter_size']
        n_channels = imgs.shape[1]
        # Convert to NCHW
        weight = filter_bank.t().reshape(-1, n_channels, filter_size, filter_size)
        padding = (filter_size - 1) // 2  # same padding
        output = F.conv2d(imgs, weight, padding=padding)
        return output

    @staticmethod
    def _extract_image_patches(img: Tensor, filter_size, stride=1, remove_mean=True, dim=0):
        # extract patches
        # skip channal dimension
        X = (img.unfold(dim + 1, filter_size, stride)
             .unfold(dim + 2, filter_size, stride))

        # unroll patches to channel-wise vectors
        X = X.flatten(dim + 1, dim + 2).flatten(-2)

        if remove_mean:
            X -= X.mean(dim=dim + 2, keepdim=True)

        # patch vectors are vertically stacked according to paper
        X = X.transpose(-1, -2)

        # Concatenate patch vectors with the same channel
        X = X.flatten(dim, dim + 1)

        return X

    @staticmethod
    def conv_output_size(w, filter_size, padding=0, stride=1):
        return int((w - filter_size + 2 * padding) / stride + 1)

    def _first_stage(self, imgs: Tensor, train) -> Tensor:
        # NCHW image
        assert imgs.dim() == 4 and imgs.nelement() > 0

        print('PCANet first stage...')

        if train:
            X = self._extract_image_patches(
                imgs, self.params['filter_size'], dim=1)
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
                    I_l[:, None, ...], filter_size, dim=1)  # N * k1k2 * mn
            self.W_2 = IncrementalPCA(
                self.params['num_filters'][1], self.batch_size).fit(Y).components_
        O = [self._convolution_output(I_i[:, None, ...], self.W_2) for I_i in I]
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
        B_h = self.conv_output_size(H, self.params['hist_blk_size'], stride=stride)
        B_w = self.conv_output_size(W, self.params['hist_blk_size'], stride=stride)
        feature_dims = B = B_h * B_w

        pyramid = self.params['pyramid']
        if pyramid:
            feature_dims = int(torch.pow(Tensor(pyramid), 2).sum().item())
        f = torch.empty(N, L1, feature_dims, 2 ** L2)

        for i, O_i in enumerate(O):  # N
            if (i + 1) % 100 == 0:
                print('Extracting PCANet feature of the %dth sample...' % (i + 1))
            Bhist_i = f[i]
            for l, O_i_l in enumerate(O_i):  # L1
                T_i_l = torch.sum(
                    map_weights[:, None, None] * heaviside(O_i_l), dim=0)  # H*W
                blocks = self._extract_image_patches(
                    T_i_l[None, ...], self.params['hist_blk_size'], stride, False).t()
                # torch.histc requires FloatTensor
                blocks = blocks.float()

                blkwise_fea = torch.empty(B, 2 ** L2)
                for b in range(B):
                    blkwise_fea[b] = torch.histc(
                        blocks[b], 2 ** L2, min=0, max=2 ** L2 - 1)

                if pyramid:
                    # Convert to NCHW
                    blkwise_fea = blkwise_fea.reshape(1, B_h, B_w, -1).permute(0, 3, 1, 2)
                    blkwise_fea = self._spatial_pyramid_pooling(blkwise_fea, pyramid)

                Bhist_i[l] = blkwise_fea

        return f.flatten(start_dim=1)

    def _spatial_pyramid_pooling(self, blkwise_fea, pyramid):
        img_size = blkwise_fea.shape[-1]  # Assume input images are square
        features = []
        for level in pyramid:
            window = math.ceil(img_size / level)
            pool = torch.nn.MaxPool2d(window, ceil_mode=True)
            features.append(pool(blkwise_fea).reshape(-1, level ** 2))

        return torch.cat(features, dim=1).t()

    def extract_features(self, imgs: Tensor, train=True):
        I = self._first_stage(imgs, train)
        O = self._second_stage(I, train)
        f = self._output_stage(O)
        return f
