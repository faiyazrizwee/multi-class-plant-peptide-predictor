import pandas as pd
import numpy as np
import logging
import os
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/02_extract_feat.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
DIPEPTIDES = [''.join(p) for p in itertools.product(AA_ALPHABET, repeat=2)]

def calc_aac(sequence):
    length = len(sequence)
    if length == 0:
        return {aa: 0 for aa in AA_ALPHABET}
    counts = {aa: sequence.count(aa) for aa in AA_ALPHABET}
    return {aa: count / length for aa, count in counts.items()}

def calc_dpc(sequence):
    length = len(sequence)
    dpc_counts = {dp: 0 for dp in DIPEPTIDES}
    if length < 2:
        return dpc_counts
    for i in range(length - 1):
        dp = sequence[i:i+2]
        if dp in dpc_counts:
            dpc_counts[dp] += 1
    
    total_dp = length - 1
    return {dp: count / total_dp for dp, count in dpc_counts.items()}

def process_features(input_file, output_file):
    logging.info(f"Extracting features for {input_file}...")
    df = pd.read_csv(input_file)
    
    aac_features = []
    dpc_features = []
    
    for seq in df['Sequence']:
        seq = str(seq).upper()
        aac_features.append(calc_aac(seq))
        dpc_features.append(calc_dpc(seq))
        
    aac_df = pd.DataFrame(aac_features)
    dpc_df = pd.DataFrame(dpc_features)
    
    # Combine features
    feat_df = pd.concat([aac_df, dpc_df], axis=1)
    
    # Prepend Entry and Label if they exist
    if 'Entry' in df.columns:
        feat_df.insert(0, 'Entry', df['Entry'])
    if 'Label' in df.columns:
        feat_df.insert(1, 'Label', df['Label'])
        
    feat_df.to_csv(output_file, index=False)
    logging.info(f"Saved features to {output_file}. Shape: {feat_df.shape}")

def main():
    os.makedirs('features', exist_ok=True)
    if os.path.exists('dataset/train.csv'):
        process_features('dataset/train.csv', 'features/train_features.csv')
    else:
        logging.error("dataset/train.csv not found.")
        
    if os.path.exists('dataset/test.csv'):
        process_features('dataset/test.csv', 'features/test_features.csv')
    else:
        logging.error("dataset/test.csv not found.")

if __name__ == "__main__":
    main()
