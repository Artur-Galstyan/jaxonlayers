import os
from collections import Counter

import numpy as np
import polars as pl
import tensorflow_datasets as tfds
from Bio import SeqIO


def load_proteinea_dataset():
    SPLITS = {
        "train": "solubility_training.csv",
        "validation": "solubility_validation.csv",
        "test": "solubility_testing.csv",
    }

    train_df = pl.read_csv("hf://datasets/proteinea/solubility/" + SPLITS["train"])
    validation_df = pl.read_csv(
        "hf://datasets/proteinea/solubility/" + SPLITS["validation"]
    )
    test_df = pl.read_csv("hf://datasets/proteinea/solubility/" + SPLITS["test"])

    all_df = pl.concat([train_df, validation_df, test_df])

    sequences = all_df["sequences"].to_list()
    labels = all_df["labels"].to_list()

    return sequences, labels


def parse_plmsol_fasta(fasta_path):
    sequences = []
    solubility_labels = []

    for record in SeqIO.parse(open(fasta_path), "fasta"):
        try:
            solubility_str = record.description.split(" ")[-1].split("-")[-1]
            sequences.append(str(record.seq))
            solubility_labels.append(solubility_str)
        except IndexError:
            print(f"Warning: Could not parse label from: {record.description}")
            continue

    return sequences, solubility_labels


def load_plmsol_dataset():
    base_path = "experiments/plmsol_dataset"

    files = ["train_dataset.fasta", "validation_dataset.fasta", "test_dataset.fasta"]

    all_sequences = []
    all_labels = []

    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            sequences, labels = parse_plmsol_fasta(file_path)
            all_sequences.extend(sequences)
            all_labels.extend(labels)
            print(f"Loaded {len(sequences)} sequences from {file}")
        else:
            print(f"Warning: {file_path} not found")

    return all_sequences, all_labels


def convert_plmsol_labels_to_binary(labels):
    binary_labels = []

    for label in labels:
        if label == "U":
            binary_labels.append(0)
        elif label == "0":
            binary_labels.append(0)
        elif label == "1":
            binary_labels.append(1)
        else:
            print(f"Unknown label found: {label}")
            binary_labels.append(0)

    return binary_labels


def analyze_datasets():
    print("Loading Proteinea dataset...")
    proteinea_seqs, proteinea_labels = load_proteinea_dataset()

    print("Loading PLM_Sol dataset...")
    plmsol_seqs, plmsol_string_labels = load_plmsol_dataset()
    plmsol_binary_labels = convert_plmsol_labels_to_binary(plmsol_string_labels)

    print("\n=== DATASET STATISTICS ===")
    print(f"Proteinea dataset:")
    print(f"  Total sequences: {len(proteinea_seqs)}")
    print(f"  Label distribution: {Counter(proteinea_labels)}")

    print(f"\nPLM_Sol dataset:")
    print(f"  Total sequences: {len(plmsol_seqs)}")
    print(f"  String label distribution: {Counter(plmsol_string_labels)}")
    print(f"  Binary label distribution: {Counter(plmsol_binary_labels)}")

    print("\n=== SEQUENCE OVERLAP ANALYSIS ===")
    proteinea_set = set(proteinea_seqs)
    plmsol_set = set(plmsol_seqs)

    intersection = proteinea_set.intersection(plmsol_set)

    print(f"Unique sequences in Proteinea: {len(proteinea_set)}")
    print(f"Unique sequences in PLM_Sol: {len(plmsol_set)}")
    print(f"Overlapping sequences: {len(intersection)}")
    print(
        f"Overlap with Proteinea: {len(intersection) / len(proteinea_set) * 100:.1f}%"
    )
    print(f"Overlap with PLM_Sol: {len(intersection) / len(plmsol_set) * 100:.1f}%")

    print("\n=== PLM_SOL PAPER VERIFICATION ===")
    plmsol_binary_counter = Counter(plmsol_binary_labels)
    insoluble_count = plmsol_binary_counter[0]
    soluble_count = plmsol_binary_counter[1]

    print(f"Expected from paper: 31,581 insoluble + 46,450 soluble = 78,031 total")
    print(
        f"Found: {insoluble_count} insoluble + {soluble_count} soluble = {len(plmsol_seqs)} total"
    )

    expected_total = 78031
    expected_insoluble = 31581
    expected_soluble = 46450

    total_match = abs(len(plmsol_seqs) - expected_total) < 100
    insoluble_match = abs(insoluble_count - expected_insoluble) < 100
    soluble_match = abs(soluble_count - expected_soluble) < 100

    print(f"Total count matches paper: {total_match}")
    print(f"Insoluble count matches paper: {insoluble_match}")
    print(f"Soluble count matches paper: {soluble_match}")

    print("\n=== LENGTH ANALYSIS ===")
    proteinea_lengths = [len(seq) for seq in proteinea_seqs]
    plmsol_lengths = [len(seq) for seq in plmsol_seqs]

    print(
        f"Proteinea sequence lengths: min={min(proteinea_lengths)}, max={max(proteinea_lengths)}, mean={np.mean(proteinea_lengths):.1f}"
    )
    print(
        f"PLM_Sol sequence lengths: min={min(plmsol_lengths)}, max={max(plmsol_lengths)}, mean={np.mean(plmsol_lengths):.1f}"
    )

    print("\n=== CONCLUSION ===")
    if len(intersection) > 0.8 * len(proteinea_set):
        print("✓ Datasets appear to be the same or very similar")
    elif len(intersection) > 0.3 * len(proteinea_set):
        print("⚠ Datasets have significant overlap but are not identical")
    else:
        print("✗ Datasets appear to be different")

    return {
        "proteinea_seqs": proteinea_seqs,
        "proteinea_labels": proteinea_labels,
        "plmsol_seqs": plmsol_seqs,
        "plmsol_labels": plmsol_binary_labels,
        "overlap_sequences": intersection,
    }


def create_plmsol_test_set(plmsol_seqs, plmsol_labels):
    test_sequences = []
    test_labels = []

    test_file = "plmsol_dataset/test_dataset.fasta"
    if os.path.exists(test_file):
        test_seqs, test_string_labels = parse_plmsol_fasta(test_file)
        test_labels = convert_plmsol_labels_to_binary(test_string_labels)

        print(f"PLM_Sol test set: {len(test_seqs)} sequences")
        print(f"Test label distribution: {Counter(test_labels)}")

        return test_seqs, test_labels
    else:
        print("PLM_Sol test file not found")
        return [], []


if __name__ == "__main__":
    data = analyze_datasets()

    test_seqs, test_labels = create_plmsol_test_set(
        data["plmsol_seqs"], data["plmsol_labels"]
    )

    if test_seqs:
        print(f"\n=== READY FOR MODEL TESTING ===")
        print(f"You can now test your model on {len(test_seqs)} PLM_Sol test sequences")
        print(f"Expected format: sequences as strings, labels as 0/1 integers")
