import random
import torch
from torch import Tensor
import numbers
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

from collections.abc import Sequence
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def random_rotate(image, mask, angel_range = 25):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-angel_range, angel_range])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    mask = mask.rotate(angle)
    return image, mask

def random_h_flip(image, mask, prob_threshold=0.5):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > prob_threshold:
        image = F.hflip(image)
        mask = F.hflip(mask)
    return image, mask

def random_v_flip(image, mask, prob_threshold=0.5):
    # 50%的概率应用垂直，垂直翻转。
    if random.random() > prob_threshold:
        image = F.vflip(image)
        mask = F.vflip(mask)
    return image, mask

def random_crop(image, mask, size = (384, 128)):
    # 随机对mask和image都进行crop增强
    i, j, h, w = transforms.RandomCrop.get_params(image,size)
    image = F.crop(image, i, j, h, w)
    mask = F.crop(mask, i, j, h, w)
    return image, mask

def random_erasing(image, mask,inplace=False, prob=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=[0.48145466, 0.4578275, 0.40821073]):
    # 随机对mask和image都进行crop增强
    if torch.rand(1) < prob:
        x, y, h, w, v = transforms.RandomErasing.get_params(F.to_tensor(image), scale=scale, ratio=ratio, value=value)
        image = F.erase(F.to_tensor(image), x, y, h, w, v, inplace)
        mask = F.erase(F.to_tensor(mask), x, y, h, w, torch.zeros([1]), inplace)
    return image, mask

def random_grayscale(image, mask, prob_threshold=0.5):
    # 50%的概率应用灰度转化。
    if random.random() > prob_threshold:
        num_output_channels, _, _ = F.get_dimensions(image)
        image = F.rgb_to_grayscale(image, num_output_channels=num_output_channels)

    return image, mask

def random_color_jittor(img, mask, brightness=.15,contrast=0,saturation=0,hue=0.1, prob_threshold=0.5):
    # check inputs and transform to tuple
    def _check_input(value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)
    
    # color jittor
    if random.random() > prob_threshold:
        brightness = _check_input(brightness, "brightness")
        contrast = _check_input(contrast, "contrast")
        saturation = _check_input(saturation, "saturation")
        hue = _check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
            brightness, contrast, saturation, hue
        )
        # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
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


def random_gaussian_blur(img, mask,kernel_size=(3, 5), sigma=(0.05, 1.5), prob_threshold=0.5):
    if random.random() < prob_threshold:
        return img, mask
    def _setup_size(size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size
    
    kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")

    for ks in kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

    if isinstance(sigma, numbers.Number):
        if sigma <= 0:
            raise ValueError("If sigma is a single number, it must be positive.")
        sigma = (sigma, sigma)
    elif isinstance(sigma, Sequence) and len(sigma) == 2:
        if not 0.0 < sigma[0] <= sigma[1]:
            raise ValueError("sigma values should be positive and of the form (min, max).")
    else:
        raise ValueError("sigma should be a single number or a list/tuple with length 2.")
    
    sigma = transforms.GaussianBlur.get_params(sigma[0], sigma[1])

    img = F.gaussian_blur(img, kernel_size, [sigma, sigma])

    return img,mask

def random_invert(img, mask, prob_threshold=0.5):
    if torch.rand(1).item() < prob_threshold:
        img =  F.invert(img)
    return img, mask

def random_posterize(img, mask, bits=3,prob_threshold=0.5):
    if torch.rand(1).item() < prob_threshold:
        img =  F.posterize(img, bits)
    return img, mask

def random_adjust_sharpness(img, mask, sharpness_factor=3,prob_threshold=0.5):
    if torch.rand(1).item() < prob_threshold:
        img = F.adjust_sharpness(img, sharpness_factor)
    return img, mask

def random_auto_contrast(img, mask, prob_threshold=0.5):
    if torch.rand(1).item() < prob_threshold:
        img = F.autocontrast(img)
    return img, mask

def random_equalize(img, mask, prob_threshold=0.5):
    if torch.rand(1).item() < prob_threshold:
        img = F.equalize(img)
    return img, mask

def random_apply(img, mask, transforms, prob_threshold=0.45):
    if prob_threshold > torch.rand(1):
        for t in transforms:
            img,mask = t(img,mask)

    return img,mask

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

def augmix(
        orig_img,
        mask,
        severity = 3,
        mixture_width = 3,
        chain_depth = -1,
        alpha = 1.0,
        all_ops = True,
        interpolation = InterpolationMode.BILINEAR,
        fill = None,
    ):
    def _augmentation_space(num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        s = {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if all_ops:
            s.update(
                {
                    "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                }
            )
        return s
    
    def _pil_to_tensor(img) -> Tensor:
        return F.pil_to_tensor(img)

    def _tensor_to_pil(img: Tensor):
        return F.to_pil_image(img)
    
    def _sample_dirichlet(params: Tensor) -> Tensor:
        # Must be on a separate method so that we can overwrite it in tests.
        return torch._sample_dirichlet(params)
    
    _PARAMETER_MAX = 10
    if not (1 <= severity <= _PARAMETER_MAX):
        raise ValueError(f"The severity must be between [1, {_PARAMETER_MAX}]. Got {severity} instead.")
    
    channels, height, width = F.get_dimensions(orig_img)
    if isinstance(orig_img, Tensor):
        img = orig_img
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * channels
        elif fill is not None:
            fill = [float(f) for f in fill]
    else:
        img = _pil_to_tensor(orig_img)

    op_meta = _augmentation_space(_PARAMETER_MAX, (height, width))

    orig_dims = list(img.shape)
    batch = img.view([1] * max(4 - img.ndim, 0) + orig_dims)
    batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

    # Sample the beta weights for combining the original and augmented image. To get Beta, we use a Dirichlet
    # with 2 parameters. The 1st column stores the weights of the original and the 2nd the ones of augmented image.
    m = _sample_dirichlet(
        torch.tensor([alpha, alpha], device=batch.device).expand(batch_dims[0], -1)
    )

    # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented images.
    combined_weights = _sample_dirichlet(
        torch.tensor([alpha] * mixture_width, device=batch.device).expand(batch_dims[0], -1)
    ) * m[:, 1].view([batch_dims[0], -1])

    mix = m[:, 0].view(batch_dims) * batch

    for i in range(mixture_width):
        aug = batch
        depth = chain_depth if chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
        for _ in range(depth):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = (
                float(magnitudes[torch.randint(severity, (1,), dtype=torch.long)].item())
                if magnitudes.ndim > 0
                else 0.0
            )
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            aug = _apply_op(aug, op_name, magnitude, interpolation=interpolation, fill=fill)
        mix.add_(combined_weights[:, i].view(batch_dims) * aug)
    mix = mix.view(orig_dims).to(dtype=img.dtype)

    if not isinstance(orig_img, Tensor):
        mix = _tensor_to_pil(mix)

    return mix,mask

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be a sequence of length {msg}.")


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

def random_affine(
        img,
        mask,
        degrees=(5, 15),
        translate=(0,0),
        scale=None,
        shear=10,
        interpolation=InterpolationMode.NEAREST,
        fill=0,
        center=None,
):
    if isinstance(interpolation, int):
        interpolation = F._interpolation_modes_from_int(interpolation)

    degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

    if translate is not None:
        _check_sequence_input(translate, "translate", req_sizes=(2,))
        for t in translate:
            if not (0.0 <= t <= 1.0):
                raise ValueError("translation values should be between 0 and 1")
    translate = translate

    if scale is not None:
        _check_sequence_input(scale, "scale", req_sizes=(2,))
        for s in scale:
            if s <= 0:
                raise ValueError("scale values should be positive")
    scale = scale

    if shear is not None:
        shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
    else:
        shear = shear

    interpolation = interpolation

    if fill is None:
        fill = 0
    elif not isinstance(fill, (Sequence, numbers.Number)):
        raise TypeError("Fill should be either a sequence or a number.")

    fill_mask = fill
    fill = fill

    if center is not None:
        _check_sequence_input(center, "center", req_sizes=(2,))
    
    center = center

    channels, height, width = F.get_dimensions(img)
    if isinstance(img, Tensor):
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * channels
        else:
            fill = [float(f) for f in fill]

    channels_mask, height_mask, width_mask = F.get_dimensions(mask)
    if isinstance(mask, Tensor):
        if isinstance(fill_mask, (int, float)):
            fill_mask = [float(fill_mask)] * channels_mask
        else:
            fill_mask = [float(f) for f in fill_mask]

    img_size = [width, height]  # flip for keeping BC on get_params call

    ret = transforms.RandomAffine.get_params(degrees, translate, scale, shear, img_size)

    img = F.affine(img, *ret, interpolation=interpolation, fill=fill, center=center)
    mask = F.affine(mask, *ret, interpolation=interpolation, fill=fill_mask, center=center)

    return img,mask

