import os
import shutil
import random
from tqdm import tqdm

# start in utils folder
SOURCE_DIR = "data_volumes/dataset/train_fixed"
DEST_DIR = "ct_rate_subset/sampled_train_fixed"
SAMPLE_SIZE = 150

if not os.path.exists(SOURCE_DIR):
    print(f"❌ Error: Source directory '{SOURCE_DIR}' not found!")
    exit(1)

os.makedirs(DEST_DIR, exist_ok=True)
print(f"Reading folders from {SOURCE_DIR}...")
all_items = os.listdir(SOURCE_DIR)

patient_folders = [f for f in all_items if f.startswith("train_") and os.path.isdir(os.path.join(SOURCE_DIR, f))]

total_available = len(patient_folders)
print(f"Found {total_available} patient folders.")

if total_available < SAMPLE_SIZE:
    print(f"❌ Error: You requested {SAMPLE_SIZE} folders but only have {total_available}!")
    exit(1)

# random.sample guarantees unique choices (no duplicates)
winners = random.sample(patient_folders, SAMPLE_SIZE)
print(f"Randomly selected {len(winners)} folders. Moving them now...")

count = 0
for folder_name in tqdm(winners):
    src_path = os.path.join(SOURCE_DIR, folder_name)
    dest_path = os.path.join(DEST_DIR, folder_name)
    
    try:
        shutil.move(src_path, dest_path)
        count += 1
    except Exception as e:
        print(f"Error moving {folder_name}: {e}")

print(f"\nSuccess! {count} folders moved to '{DEST_DIR}'")
print(f"The remaining {total_available - count} folders are left in '{SOURCE_DIR}' ready for deletion.")