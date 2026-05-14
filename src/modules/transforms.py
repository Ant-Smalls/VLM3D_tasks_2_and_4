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

class TransformData:
    """
    Transform pipeline
    
    Creates a 3-channel 3D multi-window approach:
    - Channel 0: Fluid/Soft Tissue window (pleural/pericardial effusion)
    - Channel 1: Calcification/Bone window (coronary arterial wall calcifications)
    - Channel 2: Generalist window (broad tissue contrast)
    
    Args: 
        spatial_size: Target spatial dimensions (D, H, W)
        pixdim: Target voxel spacing in mm 
        fluid_window: HU range (min, max) for soft tissue window
        calcium_window: HU range (min, max) for calcification window
        general_percentiles: Percentile range (lower, upper) for adaptive scaling
    """
    
    def __init__(
        self,
        spatial_size=(96, 224, 224),
        pixdim=(1.5, 1.5, 1.5),
        fluid_window=(-40, 400),
        calcium_window=(300, 1500),
        general_percentiles=(0.5, 99.5)
    ):
        self.spatial_size = spatial_size
        self.pixdim = pixdim
        self.fluid_window = fluid_window
        self.calcium_window = calcium_window
        self.general_percentiles = general_percentiles
        self.transform_pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        """
        Build MONAI preprocessing pipeline
        
        Returns:
            MONAI Compose pipeline
        """
        return Compose([
            # load and prepare raw data and standardize orientiation and spacing anatomically
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=self.pixdim, mode="bilinear"),

            # duplicate image for each window    
            CopyItemsd(
            keys=["image"], 
            times=3, 
            names=["image_fluid", "image_calcium", "image_general"]
            ),
            # fluid/soft tissue window for pleural and pericardial effusion
            ScaleIntensityRanged(
            keys=["image_fluid"], 
            a_min=self.fluid_window[0], 
            a_max=self.fluid_window[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
            ),
            # calcification window for coronary arterial wall calcification
            ScaleIntensityRanged(
            keys=["image_calcium"], 
            a_min=self.calcium_window[0], 
            a_max=self.calcium_window[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
            ),
            # generalist window based on percentile scaling
            ScaleIntensityRangePercentilesd(
            keys=["image_general"], 
            lower=self.general_percentiles[0], 
            upper=self.general_percentiles[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
            ),
            # merge windows into 3 channel image, clean up temp windowed images, remove empty space
            ConcatItemsd(keys=["image_fluid", "image_calcium", "image_general"], name="image", dim=0),
            DeleteItemsd(keys=["image_fluid", "image_calcium", "image_general"]),
            CropForegroundd(keys=["image"], source_key="image"),

            # standardize to target dimensions and convert types 
            ResizeWithPadOrCropd(keys=["image"], spatial_size=self.spatial_size),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32)
        ])
    
    def __call__(self, data_dict):
        """
        Apply full transform pipeline to input data
        
        Args:
            data_dict: Dictionary with 'image' key containing file path or tensor
            
        Returns:
            data_dict: Transformed dictionary with 3-channel volume [3, D, 224, 224]
        """
        return self.transform_pipeline(data_dict)


class AugmentData:
    """
    Augmentation pipeline for CT-RATE training and validation.
    
    Applies data augmentation in three stages:
    - Spatial: flips, rotations, translations, scaling
    - Noise/blur: Gaussian noise, smoothing, low-resolution simulation
    - Intensity: brightness and contrast adjustments
    
    Parameters based on "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"
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
        Get training augmentation pipeline
        
        Returns:
            MONAI Compose pipeline with training augmentations
        """
        return Compose([
            # spatial augmentations
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
            
            # noise, blur & artifact augmentations
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
            
            # intensity augmentations
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

def get_train_transforms(spatial_size=(96, 224, 224), pixdim=(1.5, 1.5, 1.5)):
    """
    Create training pipeline with preprocessing and augmentation
    
    Args:
        spatial_size: Target spatial dimensions (D, H, W)
        pixdim: Target voxel spacing in mm 
    
    Returns:
        MONAI Compose pipeline for training
    """
    preprocessing = TransformData(spatial_size=spatial_size, pixdim=pixdim)
    augmentation = AugmentData()
    
    return Compose([
        preprocessing.transform_pipeline,
        augmentation.get_train_augmentations()
    ])


def get_valid_transforms(spatial_size=(96, 224, 224), pixdim=(1.5, 1.5, 1.5)):
    """
    Create validation pipeline for preprocessing
    
    Args:
        spatial_size: Target spatial dimensions (D, H, W)
        pixdim: Target voxel spacing in mm 
    
    Returns:
        MONAI Compose pipeline 
    """
    return TransformData(spatial_size=spatial_size, pixdim=pixdim)