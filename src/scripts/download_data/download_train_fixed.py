from huggingface_hub import hf_hub_download 
import pandas as pd
import os

repo_id = "ibrahimhamamci/CT-RATE" 
directory_name = "dataset/train_fixed/"  # Updated to use train_fixed

# Read the labels file from the example_download_script directory
data = pd.read_csv("example_download_script/train_labels.csv")

print(f"Starting download of {len(data)} training volumes from train_fixed folder...")

for idx, name in enumerate(data["VolumeName"], 1):
    folder1 = name.split("_")[0]
    folder2 = name.split("_")[1]
    folder = folder1 + "_" + folder2
    folder3 = name.split("_")[2]
    subfolder = folder + "_" + folder3
    subfolder = directory_name + folder + "/" + subfolder
    
    print(f"[{idx}/{len(data)}] Downloading {name}...")
    
    try:
        hf_hub_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            token=os.environ.get("HF_TOKEN"),  
            subfolder=subfolder, 
            filename=name, 
            local_dir="data_volumes"
        )
        print(f"  ✓ Successfully downloaded {name}")
    except Exception as e:
        print(f"  ✗ Error downloading {name}: {str(e)}")

print("\n✓ Training data download complete!")
print("Data saved to: data_volumes/")
