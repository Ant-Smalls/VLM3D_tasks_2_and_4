# data_loader.py
# purpose: create the manifest for the patient data using the best volume for each patient, split the data into train/val/test sets
# author: Anthony Smaldore 

import os
import json
import csv
import random
import logging
import SimpleITK as sitk
import math
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

logger = logging.getLogger(__name__)

def create_subset_manifest(master_csv_path, base_data_dir, output_json_path):
    """
    Create the manifest for the patient data using the best volume for each patient
    Args:
        master_csv_path: Path to the CSV file containing the labels
        base_data_dir: Path to the data directory containing the patient data
        output_json_path: Path to the output JSON file containing the manifest
    Returns:
        None
    """

    manifest = []
    
    # collect all the patient directories in the data directory
    patient_dirs = [d for d in os.listdir(base_data_dir) 
                    if os.path.isdir(os.path.join(base_data_dir, d)) and d.startswith('train_')]
    
    logger.info(f"Found {len(patient_dirs)} patient directories")
    
    best_volumes = {}
    
    for patient_id in patient_dirs:
        # for each patient find the sub directories for the patient volumes
        patient_path = os.path.join(base_data_dir, patient_id)
        patient_volume_paths = []
        
        variant_dirs = [d for d in os.listdir(patient_path) 
                       if os.path.isdir(os.path.join(patient_path, d))]
        
        for variant_dir in variant_dirs:
            # for each sub directories for a patient collect all of the paths for the volume files
            variant_path = os.path.join(patient_path, variant_dir)
            
            volume_files = [f for f in os.listdir(variant_path) 
                          if f.endswith('.nii') or f.endswith('.nii.gz')]
            
            for volume_file in volume_files:
                full_volume_path = os.path.join(variant_path, volume_file)
                patient_volume_paths.append(full_volume_path)
        
        # for each patient select the best volume using the select_best_volume function
        if patient_volume_paths:
            try:
                best_volume_path = select_best_volume(patient_volume_paths)
                volume_filename = os.path.basename(best_volume_path).replace('.gz', '')
                best_volumes[volume_filename] = best_volume_path
                logger.debug(f"{patient_id}: selected best volume: {volume_filename}")
            except Exception as e:
                logger.error(f"ERROR: selecting best volume for {patient_id}: {e}")

    logger.info(f"Total best volumes collected: {len(best_volumes)}")
    logger.debug(f"Best volume keys: {list(best_volumes.keys())}")
    
    try:
        # map the best volumes selected to their labels from the file contianing the metadata for each volume
        with open(master_csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                volume_name = row['VolumeName']
                volume_name_base = volume_name.replace('.gz', '')
                
                if volume_name_base in best_volumes:
                    local_image_path = best_volumes[volume_name_base]
                    
                    labels = []
                    for col, val in row.items():
                        if col != 'VolumeName':
                            try:
                                labels.append(float(val))
                            except ValueError:
                                labels.append(val)
                    
                    manifest.append({
                        "image": local_image_path,
                        "label": labels
                    })
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return

    try:
        # create the manifest with the best volumes and their metadata 
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Created manifest with {len(manifest)} patients at {output_json_path}")
    except Exception as e:
        logger.error(f"Error saving manifest: {e}")

def select_best_volume(files_paths):
    """
    Select the best volume for a single patient across all variations and reconstructions
    Args:
        files_paths: List of .nii.gz file paths
    Returns:
        best_path: Path to the best volume
    """
    # if theres only a single file return it 
    if len(files_paths) == 1:
        return files_paths[0]

    metadata = []
    
    for path in files_paths:
        # for each of the patient volumes, read the metadata for image size (slice info - # of slices) 
        # and spacing (thickness of each slice and area (pixel spacing))
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
            
        size = reader.GetSize()       
        spacing = reader.GetSpacing()
            
        metadata.append({
            'path': path,
            'z_thickness': spacing[2],
            'xy_area': spacing[0] * spacing[1],
            'd_slices': size[2]
        })

    # normalize the values to be between 0 and 1 for each of the metrics
    z_norm = min_max_scale([m['z_thickness'] for m in metadata])
    xy_norm = min_max_scale([m['xy_area'] for m in metadata])
    d_norm = min_max_scale([m['d_slices'] for m in metadata])

    best_path = None
    best_score = float('inf')
    for i, m in enumerate(metadata):
        # for each patient find the best volume based on slice thickness, number of slices, and area (pixel spacing)
        # best volume will have the largest # of slices, thinnest slice, and smallest area (less pixel spacing for higher resolution)
        score = (0.33 * z_norm[i]) + (0.33 * xy_norm[i]) - (0.33 * d_norm[i])

        if score < best_score:
            best_score = score
            best_path = m['path']

    return best_path

def min_max_scale(values):
    """
    Min-Max scale the values to be between 0 and 1
    Args:
        values: List of values
    Returns:
        scaled_values: List of scaled values
    """
    min_val, max_val = min(values), max(values)
    # if the values are all the same return 0 for all of them
    if max_val == min_val:
        return [0.0 for _ in values]

    return [(val - min_val) / (max_val - min_val) for val in values]

def create_train_val_test_splits(manifest_path, output_dir, train_ratio=0.8, seed=42):
    """
    Create train/val/test splits from a manifest of patient volumesand save them to specified output directory
    
    Args:
        manifest_path: Path to subset_manifest.json
        output_dir: Directory where splits will be saved
        train_ratio: Fraction of data for training (default 0.8)
        seed: Random seed for reproducibility
    
    Saves:
        train_split.json: Training patient records
        val_split.json: Validation patient records  
        test_split.json: Test patient records
    """
    # load the full manifest of patient volumes and shuffle based on seed
    with open(manifest_path, 'r') as f:
        all_data = json.load(f)
    random.seed(seed)
    random.shuffle(all_data)
    
    # calculate the split sizes for the train, val, and test sets; split the data into the sets
    n_total = len(all_data)
    n_train = int(train_ratio * n_total)
    n_val = math.ceil((n_total - n_train) / 2)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]
    
    # save to specified output directory
    train_path = os.path.join(output_dir, 'train_split.json')
    val_path = os.path.join(output_dir, 'val_split.json')
    test_path = os.path.join(output_dir, 'test_split.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    logger.info(f"Created splits: Train = {len(train_data)}, Val = {len(val_data)}, Test = {len(test_data)}")
    logger.info(f"Saved splits to {output_dir}")

def create_stratified_kfold_splits(manifest_path, output_dir, k_folds=5, seed=42):
    """
    Create multilabel stratified k-fold splits to maintain class balance for the folds for cross valdiation
    Args:
        manifest_path: Path to subset_manifest.json
        output_dir: Directory where splits will be saved
        k: Number of folds (default 5)
        seed: Random seed for reproducibility
    Returns:
        None
    """

    with open(manifest_path, 'r') as f:
        all_data = json.load(f)
    
    # extract the 18-class label array for stratification
    labels = np.array([item['label'] for item in all_data])
    features = np.zeros((len(all_data), 1))

    mlskf = MultilabelStratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (train_index, validation_index) in enumerate(mlskf.split(features, labels)):
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # map the stratified indices to the original patient data directories 
        train_data = [all_data[i] for i in train_index]
        validation_data = [all_data[i] for i in validation_index]
        train_path = os.path.join(fold_dir, 'train_split.json')
        validation_path = os.path.join(fold_dir, 'val_split.json')

        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=4)
        with open(validation_path, 'w') as f:
            json.dump(validation_data, f, indent=4)

        logger.info(f"Created stratified fold {fold}: Train = {len(train_data)}, Validation = {len(validation_data)}")