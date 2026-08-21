import pandas as pd
from collections import defaultdict

csv_path = "src/data_processing/utils/example_download_script/train_labels.csv"
df = pd.read_csv(csv_path)

training_files = [
    "train_1064", "train_1331", "train_1501", "train_1666", "train_183", "train_2175", "train_2546",
    "train_2981", "train_324", "train_3379", "train_3579", "train_3958", "train_4164", "train_4395",
    "train_4689", "train_4813", "train_5035", "train_5259", "train_672", "train_1102", "train_1353",
    "train_1538", "train_1708", "train_1895", "train_2285", "train_2635", "train_3049", "train_3287",
    "train_3383", "train_3608", "train_3963", "train_4210", "train_4449", "train_469", "train_4824",
    "train_5048", "train_5278", "train_715", "train_1166", "train_1373", "train_157", "train_1712",
    "train_2", "train_2325", "train_2713", "train_3081", "train_3295", "train_3387", "train_3627",
    "train_3965", "train_4257", "train_4474", "train_4690", "train_4854", "train_5063", "train_5289",
    "train_733", "train_1211", "train_1382", "train_1573", "train_1764", "train_2008", "train_2439",
    "train_2729", "train_3105", "train_3298", "train_3440", "train_375", "train_4015", "train_4266",
    "train_45", "train_4707", "train_4862", "train_511", "train_5294", "train_772", "train_1257",
    "train_141", "train_1588", "train_1774", "train_2017", "train_2468", "train_283", "train_3136",
    "train_3313", "train_3445", "train_3804", "train_4035", "train_4274", "train_456", "train_4727",
    "train_4884", "train_5154", "train_5302", "train_841", "train_1269", "train_1413", "train_1603",
    "train_1775", "train_205", "train_250", "train_2840", "train_316", "train_3329", "train_346",
    "train_388", "train_4105", "train_4284", "train_4625", "train_4733", "train_4954", "train_5221",
    "train_5311", "train_920", "train_1278", "train_1444", "train_1616", "train_178", "train_2125",
    "train_2527", "train_2872", "train_3237", "train_3335", "train_3473", "train_395", "train_4106",
    "train_4355", "train_4673", "train_4796", "train_5004", "train_5245", "train_664", "train_1281",
    "train_1459", "train_165", "train_1823", "train_2147", "train_2543", "train_295", "train_3239",
    "train_3369", "train_3527", "train_3954", "train_4154", "train_4363", "train_4679", "train_4799",
    "train_502", "train_5257", "train_1233", "train_13284", "train_1441", "train_17187", "train_18177",
    "train_19511", "train_2345", "train_297", "train_4560", "train_5739", "train_6765", "train_7669",
    "train_8706", "train_9622", "train_10006", "train_11110", "train_12338", "train_14424", "train_15793",
    "train_1722", "train_19582", "train_235", "train_4036", "train_4622", "train_512", "train_5744",
    "train_6788", "train_769", "train_8707", "train_964", "train_10035", "train_1112", "train_1239",
    "train_13328", "train_1726", "train_1960", "train_2374", "train_3446", "train_4046", "train_5149",
    "train_577", "train_6806", "train_8716", "train_9722", "train_1008", "train_11156", "train_12473",
    "train_13497", "train_14462", "train_17290", "train_1833", "train_19605", "train_2392", "train_3052",
    "train_5867", "train_6810", "train_7728", "train_8765", "train_9733", "train_10086", "train_11175",
    "train_12559", "train_14522", "train_16032", "train_17292", "train_18361", "train_19616", "train_2393",
    "train_5178", "train_5879", "train_6830", "train_7735", "train_88", "train_9737", "train_10122",
    "train_11233", "train_13538", "train_14543", "train_16054", "train_17313", "train_18377", "train_19667",
    "train_2408", "train_3476", "train_4113", "train_5187", "train_591", "train_6833", "train_7785",
    "train_8842", "train_975", "train_10133", "train_11267", "train_12572", "train_13541", "train_16087",
    "train_17318", "train_18397", "train_1967", "train_242", "train_3129", "train_5208", "train_5911",
    "train_6835", "train_7812", "train_8866", "train_979", "train_10161", "train_11286", "train_12573",
    "train_13577", "train_14639", "train_17341", "train_18457", "train_19758", "train_3538", "train_4156",
    "train_5992", "train_6862", "train_7815", "train_8873", "train_9845", "train_10221", "train_11350",
    "train_12596", "train_13628", "train_14690", "train_1618", "train_1752", "train_18479", "train_19779",
    "train_3555", "train_6020", "train_6915", "train_7842", "train_8904", "train_9882", "train_10223",
    "train_11398", "train_12646", "train_13639", "train_14696", "train_16286", "train_17579", "train_18480",
    "train_19870", "train_248", "train_3212", "train_4192", "train_4722", "train_6048", "train_6991",
    "train_7851", "train_8930", "train_9885", "train_10257", "train_11427", "train_13708", "train_14722",
    "train_16363", "train_17638", "train_18518", "train_19887", "train_25", "train_3229", "train_4194",
    "train_6105", "train_7115", "train_7886", "train_8947", "train_9925", "train_10296", "train_11456",
    "train_12691", "train_14735", "train_16378", "train_18566", "train_3618", "train_4201", "train_6121",
    "train_7134", "train_7893", "train_8967", "train_993", "train_10356", "train_11519", "train_1270",
    "train_13743", "train_148", "train_16393", "train_17729", "train_18590", "train_4743", "train_6143",
    "train_7900", "train_8975", "train_10369", "train_11578", "train_1276", "train_13749", "train_14948",
    "train_16488", "train_18606", "train_3638", "train_4744", "train_6146", "train_7180", "train_7938",
    "train_9011", "train_1040", "train_1163", "train_12773", "train_13808", "train_14952", "train_16498",
    "train_18663", "train_2021", "train_3248", "train_3642", "train_4781", "train_53", "train_6182",
    "train_7197", "train_8041", "train_9053", "train_10423", "train_17757", "train_18702", "train_2027",
    "train_3250", "train_4790", "train_6232", "train_7221", "train_808", "train_907", "train_10468",
    "train_11681", "train_1279", "train_13844", "train_15055", "train_16548", "train_17785", "train_1871",
    "train_2031", "train_2698", "train_326", "train_3750", "train_5308", "train_6327", "train_7329",
    "train_8162", "train_10508", "train_11832", "train_139", "train_15068", "train_18802", "train_3799",
    "train_6339", "train_8183", "train_9232", "train_10514", "train_11900", "train_12847", "train_13949",
    "train_1510", "train_16672", "train_17813", "train_2062", "train_5339", "train_6343", "train_734",
    "train_8207", "train_9262", "train_1055", "train_11902", "train_12935", "train_13964", "train_15104",
    "train_16712", "train_17832", "train_18972", "train_2106", "train_2733", "train_3864", "train_4369",
    "train_5416", "train_6351", "train_7352", "train_8252", "train_9268", "train_10576", "train_11919",
    "train_12949", "train_13979", "train_15190", "train_16738", "train_17840", "train_19009", "train_2124",
    "train_2782", "train_3874", "train_4393", "train_4847", "train_5461", "train_6378", "train_7360",
    "train_8255", "train_928", "train_10580", "train_11926", "train_13014", "train_14031", "train_15203",
    "train_16819", "train_17852", "train_1903", "train_280", "train_5492", "train_6412", "train_7372",
    "train_8317", "train_9284", "train_10638", "train_1197", "train_13039", "train_15364", "train_16901",
    "train_17923", "train_1906", "train_3331", "train_39", "train_4421", "train_5494", "train_6421",
    "train_7378", "train_8387", "train_9354", "train_11984", "train_13076", "train_16904", "train_17968",
    "train_19092", "train_2836", "train_4434", "train_5527", "train_6425", "train_7425", "train_9367",
    "train_10773", "train_12012", "train_13126", "train_14134", "train_15483", "train_16976", "train_18013",
    "train_19094", "train_2196", "train_3343", "train_5533", "train_645", "train_7462", "train_8425",
    "train_9368", "train_10779", "train_12097", "train_13174", "train_14159", "train_155", "train_16988",
    "train_18058", "train_19130", "train_2225", "train_2849", "train_5543", "train_6611", "train_749",
    "train_8441", "train_9445", "train_10789", "train_12108", "train_13181", "train_14165", "train_15555",
    "train_18079", "train_19331", "train_2256", "train_3371", "train_4480", "train_5593", "train_7540",
    "train_8568", "train_9502", "train_10798", "train_13206", "train_14183", "train_1563", "train_17096",
    "train_18104", "train_1935", "train_2258", "train_2918", "train_5618", "train_6647", "train_756",
    "train_860", "train_953", "train_10844", "train_12188", "train_13259", "train_14305", "train_18115",
    "train_19363", "train_2922", "train_4014", "train_4531", "train_5666", "train_7602", "train_8618",
    "train_9537", "train_1094", "train_1221", "train_13262", "train_14334", "train_15707", "train_17151",
    "train_18116", "train_19435", "train_4535", "train_5054", "train_5690", "train_6731", "train_761",
    "train_8630", "train_955", "train_10976", "train_12315", "train_13274", "train_14335", "train_15716",
    "train_17152", "train_1812", "train_1951", "train_2343", "train_2967", "train_3416", "train_4016",
    "train_5736", "train_6753", "train_764", "train_8647", "train_9563"
]

# VolumeName is column 0; remaining columns are the 18 abnormality labels
abnormality_columns = df.columns[1:].tolist()

def get_abnormalities_for_training_file(train_id, df):
    """
    Collects per-file abnormality labels across all volumes that belong to a train_* folder.
    
    Args:
        train_id (str): Training folder ID (for example "train_1064").
        df (DataFrame): Label table with a "VolumeName" column and one column per abnormality.
    
    Returns:
        dict: Folder summary with volumes and present abnormalities, or None if no rows match.
    """
    # VolumeName is "train_{id}_{letter}_{n}.nii.gz"; startswith("{train_id}_") avoids train_10640
    pattern = f"{train_id}_"
    matching_rows = df[df['VolumeName'].str.startswith(pattern)]
    
    if len(matching_rows) == 0:
        return None
    
    abnormality_cols = df.columns[1:]
    
    # Mark an abnormality present if any volume in the folder has label 1
    abnormalities_present = {}
    for col in abnormality_cols:
        has_abnormality = (matching_rows[col] == 1).any()
        abnormalities_present[col] = has_abnormality
    
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
    """
    Prints and saves an abnormality summary for each folder in training_files.
    """
    results = {}
    for train_id in training_files:
        result = get_abnormalities_for_training_file(train_id, df)
        if result:
            results[train_id] = result

    print("Example: train_1064")
    print(f"Volumes: {results['train_1064']['volume_names']}")
    print(f"Abnormalities present: {results['train_1064']['abnormalities_present']}")
    print(f"Total abnormalities: {results['train_1064']['num_abnormalities']}")

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

    summary_df.to_csv('my_training_files_abnormalities_summary.csv', index=False)

if __name__ == "__main__":
    main()