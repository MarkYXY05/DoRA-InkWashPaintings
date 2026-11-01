import os
import json
import h5py
import argparse
from tqdm.auto import tqdm

def create_compact_h5(source_h5_path, target_h5_path, reference_json_path):
    print(f"Reading reference JSON from: {reference_json_path}")
    
    # 1. Extract unique image IDs from the VQA JSON
    with open(reference_json_path, 'r') as f:
        val_data = json.load(f)
        
    # Handle different JSON structures (list of dicts vs dict with keys)
    if isinstance(val_data, list):
        required_img_ids = {str(item['img_id']) for item in val_data if 'img_id' in item}
    else:
        raise ValueError("Unsupported JSON format. Expected a list of dictionaries with 'img_id'.")
        
    print(f"Found {len(required_img_ids)} unique required images.")
    
    # 2. Open source and target HDF5 files
    print(f"Opening source HDF5: {source_h5_path}")
    
    with h5py.File(source_h5_path, 'r') as source_f, h5py.File(target_h5_path, 'w') as target_f:
        missing_count = 0
        
        # 3. Copy only the required groups
        for img_id in tqdm(required_img_ids, desc="Compacting HDF5", unit="img"):
            
            # Formats in MSCOCO sometimes have zero-padding (e.g., 'COCO_val2014_000000123456')
            # If standard img_id isn't found, you might need to adapt the search key here.
            search_key = img_id 
            
            if search_key in source_f:
                # Create group in new file
                grp = target_f.create_group(search_key)
                
                # Copy all datasets inside the group
                for feature_key in source_f[search_key].keys():
                    grp.create_dataset(feature_key, data=source_f[f"{search_key}/{feature_key}"][()])
            else:
                missing_count += 1
                
    print("Compaction Complete")
    print(f"Target HDF5 saved to: {target_h5_path}")
    if missing_count > 0:
        print(f"WARNING: {missing_count} images were not found in the source HDF5.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a compact HDF5 feature file")
    parser.add_argument('--source_h5', type=str, required=True, help='Path to the massive source HDF5 file')
    parser.add_argument('--target_h5', type=str, required=True, help='Path to save the compact HDF5 file')
    parser.add_argument('--json_ref', type=str, required=True, help='JSON file containing the required img_ids (e.g., vqa_valid.json)')
    
    args = parser.parse_args()
    create_compact_h5(args.source_h5, args.target_h5, args.json_ref)