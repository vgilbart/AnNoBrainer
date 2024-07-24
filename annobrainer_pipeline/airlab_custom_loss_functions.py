# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch as th
import torch.nn.functional as F

import airlab.transformation as T
from annobrainer_pipeline.utils import renormalize


# Loss base class (standard from PyTorch)
class _PairwiseImageLoss(th.nn.modules.Module):
    def __init__(
        self,
        fixed_image,
        moving_image,
        fixed_mask=None,
        moving_mask=None,
        size_average=True,
        reduce=True,
    ):
        super(_PairwiseImageLoss, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._name = "parent"

        self._warped_moving_image = None
        self._warped_moving_mask = None
        self._weight = 1

        self._moving_image = moving_image
        self._moving_mask = moving_mask
        self._fixed_image = fixed_image
        self._fixed_mask = fixed_mask
        self._grid = None

        assert self._moving_image is not None and self._fixed_image is not None
        # TODO allow different image size for each image in the future
        assert self._moving_image.size == self._fixed_image.size
        assert self._moving_image.device == self._fixed_image.device
        assert (
            len(self._moving_image.size) == 2
            or len(self._moving_image.size) == 3
        )

        self._grid = T.utils.compute_grid(
            self._moving_image.size,
            dtype=self._moving_image.dtype,
            device=self._moving_image.device,
        )

        self._dtype = self._moving_image.dtype
        self._device = self._moving_image.device

    @property
    def name(self):
        return self._name

    def GetWarpedImage(self):
        return self._warped_moving_image[0, 0, ...].detach().cpu()

    def GetCurrentMask(self, displacement):
        """
        Computes a mask defining if pixels are warped outside the image domain, or if they fall into
        a fixed image mask or a warped moving image mask.
        return (Tensor): maks array
        """
        # exclude points which are transformed outside the image domain
        mask = th.zeros_like(
            self._fixed_image.image, dtype=th.uint8, device=self._device
        )
        for dim in range(displacement.size()[-1]):
            mask += displacement[..., dim].gt(1) + displacement[..., dim].lt(
                -1
            )

        mask = mask == 0

        # and exclude points which are masked by the warped moving and the fixed mask
        if self._moving_mask is not None:
            self._warped_moving_mask = F.grid_sample(
                self._moving_mask.image, displacement
            )
            self._warped_moving_mask = self._warped_moving_mask >= 0.5

            # if either the warped moving mask or the fixed mask is zero take zero,
            # otherwise take the value of mask
            if self._fixed_mask is not None:
                mask = th.where(
                    (
                        (self._warped_moving_mask == 0)
                        | (self._fixed_mask == 0)
                    ),
                    th.zeros_like(mask),
                    mask,
                )
            else:
                mask = th.where(
                    (self._warped_moving_mask == 0), th.zeros_like(mask), mask
                )

        return mask

    def set_loss_weight(self, weight):
        self._weight = weight

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return tensor.mean() * self._weight
        if not self._size_average and self._reduce:
            return tensor.sum() * self._weight
        if not self.reduce:
            return tensor * self._weight


class PM(_PairwiseImageLoss):
    r""" Custom loss function for landmark (Points Matching) based regularization.

    Minimize Squared Euclidean distance(d2) between two vectors fixed_points (f) and moving_points (m):
    d^2(f,m) = sum((fi-mi)^2)

    """

    def __init__(
        self,
        fixed_image,
        moving_image,
        fixed_mask=None,
        moving_mask=None,
        size_average=True,
        reduce=True,
        points=None,
    ):
        super(PM, self).__init__(
            fixed_image,
            moving_image,
            fixed_mask,
            moving_mask,
            size_average,
            reduce,
        )

        self._name = "pm"

        self.warped_moving_image = None

        self.fixed_pts = points["fixed"]
        self.moving_pts = points["moving"]

        self._weight = 100

    def forward(self, displacement):
        # compute displacement field
        displacement = self._grid + displacement
        # prepare placeholder for target
        target = th.zeros(displacement.shape[1:], dtype=th.float)
        # get fixed points
        fixed_pts = th.round(th.tensor(self.fixed_pts)).type(th.long)
        # get moving points, they should be closer to fixed points than previous iteration
        # renormalize - go from relative (-1,1) to real coordinates (img_width, img_height)
        moving_pts = [
            [
                renormalize(
                    x[1], (0, self._moving_image.image.shape[3]), (-1, 1)
                ),
                renormalize(
                    x[0], (0, self._moving_image.image.shape[2]), (-1, 1)
                ),
            ]
            for x in self.moving_pts
        ]

        moving_pts = th.tensor(moving_pts, dtype=th.float)
        # put fixed and moving points into tensor, preparation before calculating distance
        target = th.index_put_(
            target, (fixed_pts[:, 0], fixed_pts[:, 1]), moving_pts
        )
        # technical stuff
        target = th.tensor(
            target.clone().detach(), dtype=self._dtype, device=self._device
        )
        # calculate distance, step1
        value = displacement - target
        # step 2
        value = th.sum(value.pow(2), 3).squeeze(0)
        # technical stuff
        mask = th.zeros(value.shape, dtype=th.bool)
        mask = th.index_put_(
            mask, (fixed_pts[:, 0], fixed_pts[:, 1]), th.tensor([1]).bool()
        )

        # take into account only coordinates of interest
        mask = th.tensor(mask.clone().detach(), device=self._device)
        # final tensor with distance computed
        value = th.masked_select(value, mask)

        return self.return_loss(value)
