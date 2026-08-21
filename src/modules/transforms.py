# transforms.py
# purpose: preprocessing and augmentation pipeline
# author: Anthony Smaldore

import numpy as np
import torch
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Orientationd, 
    Spacingd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    CropForegroundd, 
    ResizeWithPadOrCropd,
    CopyItemsd, 
    ConcatItemsd, 
    DeleteItemsd,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandSimulateLowResolutiond,
    RandScaleIntensityd,
    RandAdjustContrastd,
    EnsureTyped
)

from modules.window_sets import (
    CALCIUM_WINDOW_HU,
    DEFAULT_GENERAL_PERCENTILES,
    FLUID_WINDOW_HU,
    KNOWN_WINDOW_SETS,
    LUNG_WINDOW_HU,
    WINDOW_SET_DEFAULT,
    WINDOW_SET_LUNG_MEDIASTINUM_BONE,
    resolve_window_set,
    window_set_from_checkpoint,
    window_set_from_hparams,
)

# Re-export so train/test/analysis use a single transforms import instead of circular dependecies
__all__ = [
    "AugmentData",
    "KNOWN_WINDOW_SETS",
    "TransformData",
    "WINDOW_SET_DEFAULT",
    "WINDOW_SET_LUNG_MEDIASTINUM_BONE",
    "apply_window_set",
    "get_train_transforms",
    "get_valid_transforms",
    "resolve_window_set",
    "window_set_from_checkpoint",
    "window_set_from_hparams",
]


def _channel2_intensity_transform(window_set, general_percentiles, lung_window):
    """
    Selects and returns the appropriate MONAI intensity transform for channel 2
    ("image_general") based on the window set.
    
    Args:
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
        general_percentiles (tuple): (lower, upper) percentiles for intensity normalization, used for the default window set.
        lung_window (tuple): (a_min, a_max) HU range for lung windowing, used when window_set is "lung_mediastinum_bone".
    
    Returns:
        Transform: An intensity normalization transform for channel 2.
    """
    if window_set == WINDOW_SET_LUNG_MEDIASTINUM_BONE:
        # Use a fixed HU window for the lung window set
        return ScaleIntensityRanged(
            keys=["image_general"],
            a_min=lung_window[0],
            a_max=lung_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    # Use percentile-based scaling for the default/general window set
    return ScaleIntensityRangePercentilesd(
        keys=["image_general"],
        lower=general_percentiles[0],
        upper=general_percentiles[1],
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )


def _window_intensity_pipeline(
    window_set=WINDOW_SET_DEFAULT,
    fluid_window=FLUID_WINDOW_HU,
    calcium_window=CALCIUM_WINDOW_HU,
    general_percentiles=DEFAULT_GENERAL_PERCENTILES,
    lung_window=LUNG_WINDOW_HU,
):
    """
    Builds a MONAI Compose pipeline that maps a 1-channel HU volume to 3 windowed channels.
    
    Args:
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
        fluid_window (tuple): (a_min, a_max) HU range for channel 0 (fluid/soft tissue).
        calcium_window (tuple): (a_min, a_max) HU range for channel 1 (calcium/bone).
        general_percentiles (tuple): (lower, upper) percentiles for intensity normalization, used for the default window set.
        lung_window (tuple): (a_min, a_max) HU range for lung windowing, used when window_set is "lung_mediastinum_bone".
    
    Returns:
        Compose: A MONAI pipeline that writes a 3-channel image under the "image" key.
    """
    window_set = resolve_window_set(window_set)
    return Compose([
        CopyItemsd(
            keys=["image"],
            times=3,
            names=["image_fluid", "image_calcium", "image_general"],
        ),
        ScaleIntensityRanged(
            keys=["image_fluid"],
            a_min=fluid_window[0],
            a_max=fluid_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["image_calcium"],
            a_min=calcium_window[0],
            a_max=calcium_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        _channel2_intensity_transform(window_set, general_percentiles, lung_window),
        ConcatItemsd(
            keys=["image_fluid", "image_calcium", "image_general"],
            name="image",
            dim=0,
        ),
        DeleteItemsd(keys=["image_fluid", "image_calcium", "image_general"]),
    ])


def apply_window_set(image, window_set=WINDOW_SET_DEFAULT):
    """
    Applies a named 3-channel HU layout to an already-loaded volume.
    
    Args:
        image (Tensor): Volume of shape [1, D, H, W] in HU.
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
    
    Returns:
        Tensor: Volume of shape [3, D, H, W] scaled to 0-1.
    """
    pipeline = _window_intensity_pipeline(window_set=window_set)
    out = pipeline({"image": image})
    return out["image"]


class TransformData:
    """
    Builds the load-and-window preprocessing pipeline for a 3-channel CT volume.
    
    Channel 0 is the fluid/soft tissue window. Channel 1 is the calcium/bone window.
    Channel 2 is the percentile generalist for "default", or the lung window
    [-1000, 400] HU for "lung_mediastinum_bone".
    
    Args:
        spatial_size (tuple): Target spatial dimensions (D, H, W).
        pixdim (tuple): Target voxel spacing in mm.
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
        fluid_window (tuple): (min, max) HU range for the soft tissue window.
        calcium_window (tuple): (min, max) HU range for the calcification window.
        general_percentiles (tuple): (lower, upper) percentiles for default channel 2.
        lung_window (tuple): (min, max) HU range for lung-mediastinum-bone channel 2.
    """

    def __init__(
        self,
        spatial_size=(96, 224, 224),
        pixdim=(1.5, 1.5, 1.5),
        window_set=WINDOW_SET_DEFAULT,
        fluid_window=FLUID_WINDOW_HU,
        calcium_window=CALCIUM_WINDOW_HU,
        general_percentiles=DEFAULT_GENERAL_PERCENTILES,
        lung_window=LUNG_WINDOW_HU,
    ):
        self.spatial_size = spatial_size
        self.pixdim = pixdim
        self.window_set = resolve_window_set(window_set)
        self.fluid_window = fluid_window
        self.calcium_window = calcium_window
        self.general_percentiles = general_percentiles
        self.lung_window = lung_window
        self.transform_pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        """
        Builds the MONAI preprocessing pipeline from load through windowing and resize.
        
        Returns:
            Compose: A MONAI preprocessing pipeline.
        """
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=self.pixdim, mode="bilinear"),

            _window_intensity_pipeline(
                window_set=self.window_set,
                fluid_window=self.fluid_window,
                calcium_window=self.calcium_window,
                general_percentiles=self.general_percentiles,
                lung_window=self.lung_window,
            ),
            CropForegroundd(keys=["image"], source_key="image"),

            ResizeWithPadOrCropd(keys=["image"], spatial_size=self.spatial_size),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32)
        ])
    
    def __call__(self, data_dict):
        """
        Applies the full transform pipeline to an input data dictionary.
        
        Args:
            data_dict (dict): Dictionary with an "image" key containing a file path or tensor.
        
        Returns:
            dict: Transformed dictionary with a 3-channel volume of shape [3, D, H, W].
        """
        return self.transform_pipeline(data_dict)


class AugmentData:
    """
    Builds the CT-RATE training augmentation pipeline.
    
    Augmentations run in three stages: spatial (flips, rotations, translations, scaling),
    noise/blur (Gaussian noise, smoothing, low-resolution simulation), and intensity
    (brightness and contrast). Parameters follow "Revisiting 2D Foundation Models for
    Scalable 3D Medical Image Classification".
    """
    
    def __init__(self):
        self.spatial_flip_prob = 0.5
        self.spatial_affine_prob = 0.2
        self.spatial_rotate_range = (np.pi/6, np.pi/6, np.pi/6)
        self.spatial_translate_range = (10, 10, 10)
        self.spatial_scale_range = (0.3, 0.3, 0.3)
        
        self.noise_gaussian_prob = 0.25
        self.noise_gaussian_std = 0.1
        self.noise_smooth_prob = 0.2
        self.noise_smooth_sigma = (0.5, 1.0)
        self.noise_lowres_prob = 0.2
        self.noise_lowres_zoom = (0.5, 1.0)
        
        self.intensity_scale_prob = 0.15
        self.intensity_scale_factors = 0.25
        self.intensity_contrast_prob = 0.25
        self.intensity_contrast_gamma = (0.7, 1.5)
    
    def get_train_augmentations(self):
        """
        Returns the training augmentation pipeline.
        
        Returns:
            Compose: A MONAI pipeline with spatial, noise, and intensity augmentations.
        """
        return Compose([
            # Spatial augmentations
            RandFlipd(
                keys=["image"], 
                prob=self.spatial_flip_prob, 
                spatial_axis=[0, 1, 2]
            ),
            RandAffined(
                keys=["image"], 
                prob=self.spatial_affine_prob, 
                rotate_range=self.spatial_rotate_range,
                translate_range=self.spatial_translate_range,
                scale_range=self.spatial_scale_range,
                mode="bilinear",
                padding_mode="zeros"
            ),
            
            # Noise, blur, and artifact augmentations
            RandGaussianNoised(
                keys=["image"], 
                prob=self.noise_gaussian_prob, 
                mean=0.0, 
                std=self.noise_gaussian_std
            ),
            RandGaussianSmoothd(
                keys=["image"], 
                prob=self.noise_smooth_prob, 
                sigma_x=self.noise_smooth_sigma, 
                sigma_y=self.noise_smooth_sigma, 
                sigma_z=self.noise_smooth_sigma
            ),
            RandSimulateLowResolutiond(
                keys=["image"], 
                prob=self.noise_lowres_prob, 
                zoom_range=self.noise_lowres_zoom
            ),
            
            # Intensity augmentations
            RandScaleIntensityd(
                keys=["image"], 
                factors=self.intensity_scale_factors, 
                prob=self.intensity_scale_prob
            ),
            RandAdjustContrastd(
                keys=["image"], 
                gamma=self.intensity_contrast_gamma, 
                prob=self.intensity_contrast_prob
            )
        ])

def get_train_transforms(
    spatial_size=(96, 224, 224),
    pixdim=(1.5, 1.5, 1.5),
    window_set=WINDOW_SET_DEFAULT,
):
    """
    Creates the training pipeline with preprocessing and augmentation.
    
    Args:
        spatial_size (tuple): Target spatial dimensions (D, H, W).
        pixdim (tuple): Target voxel spacing in mm.
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
    
    Returns:
        Compose: A MONAI pipeline for training.
    """
    preprocessing = TransformData(
        spatial_size=spatial_size,
        pixdim=pixdim,
        window_set=window_set,
    )
    augmentation = AugmentData()
    
    return Compose([
        preprocessing.transform_pipeline,
        augmentation.get_train_augmentations()
    ])


def get_valid_transforms(
    spatial_size=(96, 224, 224),
    pixdim=(1.5, 1.5, 1.5),
    window_set=WINDOW_SET_DEFAULT,
):
    """
    Creates the validation pipeline with preprocessing only.
    
    Args:
        spatial_size (tuple): Target spatial dimensions (D, H, W).
        pixdim (tuple): Target voxel spacing in mm.
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
    
    Returns:
        TransformData: A callable preprocessing pipeline for validation and test.
    """
    return TransformData(
        spatial_size=spatial_size,
        pixdim=pixdim,
        window_set=window_set,
    )