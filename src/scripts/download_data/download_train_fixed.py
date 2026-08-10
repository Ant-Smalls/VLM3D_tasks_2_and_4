from huggingface_hub import hf_hub_download 
import pandas as pd
import os
import random 
import argparse

def download_training_data(output_dir, num_folders, summary_path, labels_path):
    repo_id = "ibrahimhamamci/CT-RATE" 
    directory_name = "dataset/train_fixed/"
    
    # 1. Read the summary file to get the list of folders to exclude
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        excluded_folders = set(summary_df["TrainingFile"])
        print(f"Found {len(excluded_folders)} folders to exclude from {summary_path}.")
    else:
        excluded_folders = set()
        print(f"Warning: Could not find {summary_path}. No folders will be excluded.")
        
    # 2. Define the pool of available folders (train_1 to train_20000)
    all_possible_folders = set(f"train_{i}" for i in range(1, 20001))
    
    # Remove the excluded folders from our available pool
    available_folders = list(all_possible_folders - excluded_folders)
    
    if not available_folders:
        print("No available folders left to download!")
        return
        
    # 3. Randomly sample the desired number of folders
    num_to_sample = min(num_folders, len(available_folders))
    selected_folders = set(random.sample(available_folders, num_to_sample))
    print(f"Randomly selected {num_to_sample} new folders to download.")
    
    # 4. Read the labels file
    if not os.path.exists(labels_path):
        print(f"Error: Could not find labels file at {labels_path}")
        return
        
    data = pd.read_csv(labels_path)
    
    print("Starting download...")
    
    download_count = 0
    for idx, name in enumerate(data["VolumeName"], 1):
        # Extract the folder name (e.g., "train_1" from "train_1_a_1.nii.gz")
        parts = name.split("_")
        folder = parts[0] + "_" + parts[1]
        
        # Skip if the folder is NOT in our randomly selected set
        if folder not in selected_folders:
            continue
            
        folder3 = parts[2]
        subfolder = folder + "_" + folder3
        subfolder = directory_name + folder + "/" + subfolder
        
        download_count += 1
        print(f"[{download_count}] Downloading {name} (from folder {folder})...")
        
        try:
            hf_hub_download(
                repo_id=repo_id, 
                repo_type="dataset", 
                token=os.environ.get("HF_TOKEN"),  
                subfolder=subfolder, 
                filename=name, 
                local_dir=output_dir
            )
            print(f"  ✓ Successfully downloaded {name}")
        except Exception as e:
            print(f"  ✗ Error downloading {name}: {str(e)}")
    print(f"\n✓ Download complete! Downloaded {download_count} volumes across {num_to_sample} folders.")
    print(f"Data saved to: {output_dir}/")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CT-RATE training data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save downloaded data")
    parser.add_argument("--num_folders", type=int, default=100, help="Number of random folders to download (default: 100)")
    parser.add_argument("--summary_path", type=str, default="my_training_files_abnormalities_summary.csv", help="Path to the summary CSV file of already downloaded folders")
    parser.add_argument("--labels_path", type=str, default="src/scripts/download_data/train_data_labels/train_labels.csv", help="Path to the train_labels.csv file")
    args = parser.parse_args()
    
    download_training_data(args.output_dir, args.num_folders, args.summary_path, args.labels_path)
