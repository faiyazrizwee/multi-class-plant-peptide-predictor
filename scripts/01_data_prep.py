import os
import pandas as pd
import random
import subprocess
import glob

# Set fixed seed for reproducibility
random_seed = 42
random.seed(random_seed)

print("Starting Capped (500) CD-HIT Cluster Split Pipeline...")

# Known family CSV files
expected_families = ['CLE', 'Cyclotides', 'Defensins', 'LTPs', 'PSKs', 'RALFs', 'Snakins', 'Thionins', 'Other']

# Setup directories
base_dir = '/home/faiyaz/multi_class_pep_capped_cluster_80_20'
train_dir = os.path.join(base_dir, 'dataset', 'Train')
test_dir = os.path.join(base_dir, 'dataset', 'Test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Prepare lists for merged datasets and summary
all_train_df = []
all_test_df = []
summary_data = []

# Process each family
for family in expected_families:
    csv_path = os.path.join(base_dir, f"{family}.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping.")
        continue
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        continue
        
    original_len = len(df)
    
    # ----------------------------------------------------
    # STEP 2: Downsample to exactly 500 if larger (EXCEPT Snakins)
    # ----------------------------------------------------
    if original_len > 500 and family != 'Snakins':
        df = df.sample(n=500, random_state=random_seed)
    selected_len = len(df)
    
    # Sort the index to preserve original row order after sampling
    df = df.sort_index()
    
    # Reset index to easily map back to the dataframe after CD-HIT
    df = df.reset_index(drop=True)
    
    # ----------------------------------------------------
    # CD-HIT CLUSTERING (80/20 Split)
    # ----------------------------------------------------
    fasta_path = os.path.join(base_dir, f"temp_{family}.fasta")
    clstr_path = os.path.join(base_dir, f"temp_{family}_clustered.fasta")
    
    # Write fasta
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{idx}\n{row['Sequence']}\n")
            
    # Run CD-HIT at 50% identity
    # cd-hit -i temp.fasta -o temp_clustered.fasta -c 0.5 -n 2 -d 0 -M 0 -T 0
    cmd = ["cd-hit", "-i", fasta_path, "-o", clstr_path, "-c", "0.5", "-n", "2", "-d", "0", "-M", "0", "-T", "0"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Parse .clstr file
    clusters = []
    current_cluster = []
    with open(clstr_path + ".clstr", 'r') as f:
        for line in f:
            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
            else:
                # Format: 0	67aa, >141... *
                parts = line.strip().split()
                if len(parts) > 2:
                    idx_str = parts[2].replace(">", "").replace("...", "")
                    current_cluster.append(int(idx_str))
    if current_cluster:
        clusters.append(current_cluster)
        
    # Clean up temp files
    if os.path.exists(fasta_path): os.remove(fasta_path)
    if os.path.exists(clstr_path): os.remove(clstr_path)
    if os.path.exists(clstr_path + ".clstr"): os.remove(clstr_path + ".clstr")
    
    # Shuffle clusters for random Train/Test assignment
    random.seed(random_seed)
    random.shuffle(clusters)
    
    # Sort clusters largest to smallest for greedy packing
    clusters.sort(key=len, reverse=True)
    
    train_indices = []
    test_indices = []
    
    # Target 80% train
    target_train_size = int(0.8 * len(df))
    
    for cluster in clusters:
        # If we haven't reached the target train size, try to add it
        if len(train_indices) < target_train_size:
            # Only add if it doesn't drastically overshoot the 80% target
            # We allow a small overshoot buffer (e.g., +20 sequences)
            if len(train_indices) + len(cluster) <= target_train_size + 20:
                train_indices.extend(cluster)
            else:
                # If adding it would overshoot significantly, push it to the test set
                test_indices.extend(cluster)
        else:
            test_indices.extend(cluster)
            
    # Map back to original rows and sort index to preserve sequence order
    train_df = df.iloc[train_indices].sort_index()
    test_df = df.iloc[test_indices].sort_index()
    
    # Ensure the label column exists (add if missing)
    if 'Label' not in train_df.columns:
        train_df['Label'] = family
        test_df['Label'] = family
    
    # Save individual class files in Dataset/Train/ and Dataset/Test/
    train_out_path = os.path.join(train_dir, f"{family}.csv")
    test_out_path = os.path.join(test_dir, f"{family}.csv")
    
    train_df.to_csv(train_out_path, index=False)
    test_df.to_csv(test_out_path, index=False)
    
    all_train_df.append(train_df)
    all_test_df.append(test_df)
    
    summary_data.append({
        'Family': family,
        'Original sequences': original_len,
        'Selected sequences': selected_len,
        'Training sequences': len(train_df),
        'Testing sequences': len(test_df)
    })

# Compile global train/test datasets
final_train = pd.concat(all_train_df, ignore_index=True)
final_test = pd.concat(all_test_df, ignore_index=True)

final_train.to_csv(os.path.join(base_dir, 'dataset', 'train.csv'), index=False)
final_test.to_csv(os.path.join(base_dir, 'dataset', 'test.csv'), index=False)
final_train.to_csv(os.path.join(base_dir, 'dataset', 'full_dataset.csv'), index=False) # Overwrite full dataset just in case downstream relies on it (we can skip combining Train/Test since train/test split is strictly defined now)
# Wait, actually full_dataset.csv is just for metadata purposes sometimes. Let's make it the true full capped dataset.
final_full = pd.concat([final_train, final_test], ignore_index=True)
final_full.to_csv(os.path.join(base_dir, 'dataset', 'full_dataset.csv'), index=False)

# Save and print summary
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(base_dir, 'dataset_summary.csv'), index=False)

print("\n=======================================================")
print("                   DATASET SUMMARY                     ")
print("=======================================================")
print(summary_df.to_string(index=False))
print("=======================================================")
print(f"Total Train: {len(final_train)} | Total Test: {len(final_test)}")
print("Data preparation complete. All splits saved independently.")
