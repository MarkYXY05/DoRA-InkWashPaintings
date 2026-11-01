import sys
import csv
import base64
import time
import argparse
import numpy as np
import h5py
from tqdm.auto import tqdm

# Increase CSV field size limit to handle large base64 strings
csv.field_size_limit(sys.maxsize)

FIELDNAMES = [
    "img_id", "img_h", "img_w", "objects_id", "objects_conf",
    "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"
]

def decode_base64_tensor(encoded_str, shape, dtype):
    """Decodes a base64 string back into a numpy array."""
    decoded_bytes = base64.b64decode(encoded_str)
    tensor = np.frombuffer(decoded_bytes, dtype=dtype)
    return tensor.reshape(shape)

def convert_tsv_to_hdf5(tsv_path, h5_path):
    print(f"Starting conversion: [{tsv_path}] -> [{h5_path}]")
    start_time = time.time()
    
    # Open TSV for reading and HDF5 for writing simultaneously
    with open(tsv_path, 'r') as tsv_file, h5py.File(h5_path, 'w') as h5_file:
        reader = csv.DictReader(tsv_file, FIELDNAMES, delimiter="\t")
        
        for item in tqdm(reader, desc="Processing Images", unit="img"):
            img_id = item['img_id']
            num_boxes = int(item['num_boxes'])
            img_h = int(item['img_h'])
            img_w = int(item['img_w'])
            
            # Decoding configuration
            decode_config = [
                ('objects_id', (num_boxes,), np.int64),
                ('objects_conf', (num_boxes,), np.float32),
                ('attrs_id', (num_boxes,), np.int64),
                ('attrs_conf', (num_boxes,), np.float32),
                ('boxes', (num_boxes, 4), np.float32),
                ('features', (num_boxes, 2048), np.float32),
            ]
            
            # Create a group for this image in the HDF5 file
            grp = h5_file.create_group(img_id)
            grp['img_h'] = img_h
            grp['img_w'] = img_w
            
            # Decode each field and save directly to HDF5
            for key, shape, dtype in decode_config:
                tensor = decode_base64_tensor(item[key], shape, dtype)
                grp.create_dataset(key, data=tensor)
                
    elapsed_time = time.time() - start_time
    print(f"Success! Conversion completed in {elapsed_time:.2f} seconds.")
    print(f"Saved HDF5 file to: {h5_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert TSV features to HDF5")
    parser.add_argument('--tsv_path', type=str, required=True, help='Input TSV file path')
    parser.add_argument('--h5_path', type=str, required=True, help='Output HDF5 file path')
    
    args = parser.parse_args()
    convert_tsv_to_hdf5(args.tsv_path, args.h5_path)