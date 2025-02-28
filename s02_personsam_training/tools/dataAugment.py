import random
import torch
from torch import Tensor
import numbers
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple
from torchvision import transforms

from collections.abc import Sequence
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class ImageMaskRandomRotation(transforms.RandomRotation):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        img = F.rotate(img, angle, self.interpolation, self.expand, self.center, fill)
        mask = F.rotate(mask, angle, self.interpolation, self.expand, self.center, fill)
        return img,mask
    
class ImageMaskRandomHFlip(transforms.RandomHorizontalFlip):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img,mask
    
class ImageMaskRandomPerspective(transforms.RandomPerspective):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """

        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        fill_mask = self.fill
        channels_mask, height, width = F.get_dimensions(mask)
        if isinstance(mask, Tensor):
            if isinstance(fill_mask, (int, float)):
                fill_mask = [float(fill_mask)] * channels_mask
            else:
                fill_mask = [float(f) for f in fill_mask]

        if torch.rand(1) < self.p:
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            img = F.perspective(img, startpoints, endpoints, self.interpolation, fill)
            mask = F.perspective(mask, startpoints, endpoints, self.interpolation, fill_mask)
        return img,mask
    

class ImageMaskColorJitter(transforms.ColorJitter):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img,mask
    
class ImageMaskRandomAffine(transforms.RandomAffine):
    def forward(self, img, mask):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        fill_mask = self.fill
        channels_mask, height, width = F.get_dimensions(mask)
        if isinstance(mask, Tensor):
            if isinstance(fill_mask, (int, float)):
                fill_mask = [float(fill_mask)] * channels_mask
            else:
                fill_mask = [float(f) for f in fill_mask]

        img_size = [width, height]  # flip for keeping BC on get_params call

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        img = F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
        mask = F.affine(mask, *ret, interpolation=self.interpolation, fill=fill_mask, center=self.center)
        
        return img, mask
    
class ImageMaskRandomGrayscale(transforms.RandomGrayscale):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        num_output_channels, _, _ = F.get_dimensions(img)
        if torch.rand(1) < self.p:
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        return img, mask
    
class ImageMaskRandomErasing(transforms.RandomErasing):
    def forward(self, img, mask):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:
            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value

            # if value is not None and not (len(value) in (1, img.shape[-3])):
            #     raise ValueError(
            #         "If value is a sequence, it should have either a single value or "
            #         f"{img.shape[-3]} (number of input channels)"
            #     )

            x, y, h, w, v = self.get_params(F.to_tensor(img), scale=self.scale, ratio=self.ratio, value=value)
            mask = F.erase(F.to_tensor(mask), x, y, h, w, torch.zeros([1]), self.inplace)
            img = F.erase(F.to_tensor(img), x, y, h, w, v, self.inplace)
        return img, mask
    
class ImageMaskGaussianBlur(transforms.GaussianBlur):
    def forward(self, img: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        img = F.gaussian_blur(img, self.kernel_size, [sigma, sigma])

        return img, mask

class ImageMaskRandomInvert(transforms.RandomInvert):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be inverted.

        Returns:
            PIL Image or Tensor: Randomly color inverted image.
        """
        if torch.rand(1).item() < self.p:
            img = F.invert(img)
        return img, mask

class ImageMaskRandomPosterize(transforms.RandomPosterize):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be posterized.

        Returns:
            PIL Image or Tensor: Randomly posterized image.
        """
        if torch.rand(1).item() < self.p:
            img = F.posterize(img, self.bits)
        return img, mask

class ImageMaskRandomSolarize(transforms.RandomSolarize):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be solarized.

        Returns:
            PIL Image or Tensor: Randomly solarized image.
        """
        if torch.rand(1).item() < self.p:
            img = F.solarize(img, self.threshold)
        return img, mask

class ImageMaskRandomAdjustSharpness(transforms.RandomAdjustSharpness):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be sharpened.

        Returns:
            PIL Image or Tensor: Randomly sharpened image.
        """
        if torch.rand(1).item() < self.p:
            img = F.adjust_sharpness(img, self.sharpness_factor)
        return img, mask

class ImageMaskRandomAutocontrast(transforms.RandomAutocontrast):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be autocontrasted.

        Returns:
            PIL Image or Tensor: Randomly autocontrasted image.
        """
        if torch.rand(1).item() < self.p:
            img = F.autocontrast(img)
        return img,mask
    
class ImageMaskRandomEqualize(transforms.RandomEqualize):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be equalized.

        Returns:
            PIL Image or Tensor: Randomly equalized image.
        """
        if torch.rand(1).item() < self.p:
            img = F.equalize(img)
        return img, mask

class ImageMaskRandomCrop(transforms.RandomCrop):
    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return img, mask

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

class ImageMaskAugMix(transforms.AugMix):
    def forward(self, orig_img: Tensor, mask: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(orig_img)
        if isinstance(orig_img, Tensor):
            img = orig_img
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]
        else:
            img = self._pil_to_tensor(orig_img)

        op_meta = self._augmentation_space(self._PARAMETER_MAX, (height, width))

        orig_dims = list(img.shape)
        batch = img.view([1] * max(4 - img.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

        # Sample the beta weights for combining the original and augmented image. To get Beta, we use a Dirichlet
        # with 2 parameters. The 1st column stores the weights of the original and the 2nd the ones of augmented image.
        m = self._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1)
        )

        # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented images.
        combined_weights = self._sample_dirichlet(
            torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)
        ) * m[:, 1].view([batch_dims[0], -1])

        mix = m[:, 0].view(batch_dims) * batch
        for i in range(self.mixture_width):
            aug = batch
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            for _ in range(depth):
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[torch.randint(self.severity, (1,), dtype=torch.long)].item())
                    if magnitudes.ndim > 0
                    else 0.0
                )
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                aug = _apply_op(aug, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            mix.add_(combined_weights[:, i].view(batch_dims) * aug)
        mix = mix.view(orig_dims).to(dtype=img.dtype)

        if not isinstance(orig_img, Tensor):
            mix = self._tensor_to_pil(mix)
        return mix, mask
    
class ImageMaskRandomApply(transforms.RandomApply):
    def forward(self, img, mask):
        if self.p < torch.rand(1):
            return img, mask
        for t in self.transforms:
            img,mask = t(img,mask)
        return img, mask
    
# class ImageMaskToTensor(transforms.ToTensor):
#     def __call__(self, pic, mask):
#         """
#         Args:
#             pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

#         Returns:
#             Tensor: Converted image.
#         """
#         return F.to_tensor(pic), F.to_tensor(mask)