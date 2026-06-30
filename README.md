# Multi-class Peptide Family Predictor

This repository contains a standalone command-line tool (`predict.py`) to classify peptide sequences into one of 9 distinct families (e.g., CLE, Cyclotides, Defensins, LTPs, PSKs, RALFs, Snakins, Thionins, Other). 

The predictor uses a trained LightGBM machine learning model based on Amino Acid Composition (AAC) and Dipeptide Composition (DPC) features.

## Prerequisites

Make sure you have Python 3 installed along with the required libraries.

```bash
pip install pandas numpy scikit-learn lightgbm joblib
```

*Note: The model files must exist in the `models/` directory (`best_classical_model.pkl`, `scaler.pkl`, `label_encoder.pkl`).*

## How to Run

You can run the predictor using the `predict.py` script. It accepts either a single sequence or a FASTA file containing multiple sequences.

### 1. Predict a Single Sequence

Use the `-s` or `--sequence` flag to pass a single peptide string.

```bash
python predict.py -s "MAQSLTLIFVILILGLASLASSARAEKQLAEKAAAKLAEKAAAKLAEKAA"
```

**Output:**
```
Sequence: MAQSLTLIFVILILGLASLASSARAEKQLAEKAAAKLAEKAAAKLAEKAA
Class                Prediction Score
----------------------------------------
CLE                  0.902
LTPs                 0.051
Thionins             0.012
```

### 2. Predict from a FASTA File

Use the `-f` or `--fasta` flag to process multiple sequences from a FASTA file.

```bash
python predict.py -f dataset/temp_Thionins.fasta
```

### 3. Save Results to a CSV File

You can use the `-o` or `--output` flag to save the top 3 predictions for your sequence(s) directly to a CSV file.

```bash
python predict.py -s "MAQSLTLIFVILILGLASLASSARAEKQLAEKAAAKLAEKAAAKLAEKAA" -o results.csv
```

### Help Command

To see all available options, run:

```bash
python predict.py --help
```

## Available Model Families
The model is capable of classifying peptides into the following families:
- CLE
- Cyclotides
- Defensins
- LTPs
- PSKs
- RALFs
- Snakins
- Thionins
- Other
