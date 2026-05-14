import pandas as pd
from collections import defaultdict

# Read the CSV file
csv_path = "src/data_processing/utils/example_download_script/train_labels.csv"
df = pd.read_csv(csv_path)

# My training files
training_files = [
    "train_1064", "train_1331", "train_1501", "train_1666", "train_183", "train_2175", 
    "train_2546", "train_2981", "train_324", "train_3379", "train_3579", "train_3958", 
    "train_4164", "train_4395", "train_4689", "train_4813", "train_5035", "train_5259", "train_672",
    "train_1102", "train_1353", "train_1538", "train_1708", "train_1895", "train_2285", 
    "train_2635", "train_3049", "train_3287", "train_3383", "train_3608", "train_3963", 
    "train_4210", "train_4449", "train_469", "train_4824", "train_5048", "train_5278", "train_715",
    "train_1166", "train_1373", "train_157", "train_1712", "train_2", "train_2325", 
    "train_2713", "train_3081", "train_3295", "train_3387", "train_3627", "train_3965", 
    "train_4257", "train_4474", "train_4690", "train_4854", "train_5063", "train_5289", "train_733",
    "train_1211", "train_1382", "train_1573", "train_1764", "train_2008", "train_2439", 
    "train_2729", "train_3105", "train_3298", "train_3440", "train_375", "train_4015", 
    "train_4266", "train_45", "train_4707", "train_4862", "train_511", "train_5294", "train_772",
    "train_1257", "train_141", "train_1588", "train_1774", "train_2017", "train_2468", 
    "train_283", "train_3136", "train_3313", "train_3445", "train_3804", "train_4035", 
    "train_4274", "train_456", "train_4727", "train_4884", "train_5154", "train_5302", "train_841",
    "train_1269", "train_1413", "train_1603", "train_1775", "train_205", "train_250", 
    "train_2840", "train_316", "train_3329", "train_346", "train_388", "train_4105", 
    "train_4284", "train_4625", "train_4733", "train_4954", "train_5221", "train_5311", "train_920",
    "train_1278", "train_1444", "train_1616", "train_178", "train_2125", "train_2527", 
    "train_2872", "train_3237", "train_3335", "train_3473", "train_395", "train_4106", 
    "train_4355", "train_4673", "train_4796", "train_5004", "train_5245", "train_664",
    "train_1281", "train_1459", "train_165", "train_1823", "train_2147", "train_2543", 
    "train_295", "train_3239", "train_3369", "train_3527", "train_3954", "train_4154", 
    "train_4363", "train_4679", "train_4799", "train_502", "train_5257", "train_671"
]

# Get all abnormality columns (excluding VolumeName)
abnormality_columns = df.columns[1:].tolist()

# Create a mapping for each training file
def get_abnormalities_for_training_file(train_id, df):
    """
    Get all abnormalities present for a given training file ID.
    
    Args:
        train_id: Training file ID (e.g., "train_1064")
        df: DataFrame with the labels
    
    Returns:
        Dictionary with abnormality information
    """
    # Filter rows that match this training ID
    # Pattern: train_1064_a_1.nii.gz, train_1064_a_2.nii.gz, train_1064_b_1.nii.gz, etc.
    pattern = f"{train_id}_"
    matching_rows = df[df['VolumeName'].str.startswith(pattern)]
    
    if len(matching_rows) == 0:
        return None
    
    # Get abnormality columns
    abnormality_cols = df.columns[1:]
    
    # Aggregate: if ANY volume has the abnormality (value = 1), mark it as present
    abnormalities_present = {}
    for col in abnormality_cols:
        has_abnormality = (matching_rows[col] == 1).any()
        abnormalities_present[col] = has_abnormality
    
    # Get list of present abnormalities
    present_list = [col for col, present in abnormalities_present.items() if present]
    
    return {
        'train_id': train_id,
        'num_volumes': len(matching_rows),
        'volume_names': matching_rows['VolumeName'].tolist(),
        'abnormalities_dict': abnormalities_present,
        'abnormalities_present': present_list,
        'num_abnormalities': len(present_list)
    }

def main():
    # Map all your training files
    results = {}
    for train_id in training_files:
        result = get_abnormalities_for_training_file(train_id, df)
        if result:
            results[train_id] = result

    # Example: Print results for a specific training file
    print("Example: train_1064")
    print(f"Volumes: {results['train_1064']['volume_names']}")
    print(f"Abnormalities present: {results['train_1064']['abnormalities_present']}")
    print(f"Total abnormalities: {results['train_1064']['num_abnormalities']}")

    # Create a summary DataFrame
    summary_data = []
    for train_id, info in results.items():
        summary_data.append({
            'TrainingFile': train_id,
            'NumVolumes': info['num_volumes'],
            'NumAbnormalities': info['num_abnormalities'],
            'Abnormalities': ', '.join(info['abnormalities_present'])
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("SUMMARY OF ALL TRAINING FILES")
    print("="*80)
    print(summary_df.to_string(index=False))

    # Save to CSV if needed
    summary_df.to_csv('my_training_files_abnormalities_summary.csv', index=False)

if __name__ == "__main__":
    main()