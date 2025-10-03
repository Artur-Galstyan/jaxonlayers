import json
import os

import array_record
import grain
import jax.numpy as jnp
import polars as pl
from Bio import SeqIO

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = ["Z"] + list(AMINO_ACIDS)
INT_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}
CHAR_TO_INT = {char: i for i, char in enumerate(VOCAB)}


class ProteinPreprocessor(grain.transforms.Map):
    max_protein_length: int

    def __init__(self, max_protein_length: int):
        self.max_protein_length = max_protein_length

    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        sequence = sequence[: self.max_protein_length]
        sequence = sequence.ljust(self.max_protein_length, "Z")
        indices = [CHAR_TO_INT.get(aa, 0) for aa in sequence]
        return {
            "features": jnp.array(indices, jnp.int32),
            "sequence": sequence,
            "label": jnp.zeros(shape=(2,)).at[label].set(1),
        }


class DataFrameDataSource:
    def __init__(self, df: pl.DataFrame):
        self._data = df.to_dicts()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, record_key: int) -> dict:
        return self._data[record_key]


def parse_plmsol_fasta(fasta_path):
    sequences = []
    solubility_labels = []
    if not os.path.exists(fasta_path):
        print(f"Warning: Fasta file not found at {fasta_path}")
        return sequences, solubility_labels
    for record in SeqIO.parse(open(fasta_path), "fasta"):
        try:
            solubility_str = record.description.split(" ")[-1].split("-")[-1]
            sequences.append(str(record.seq))
            solubility_labels.append(solubility_str)
        except IndexError:
            print(f"Warning: Could not parse label from: {record.description}")
            continue
    return sequences, solubility_labels


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
            binary_labels.append(0)
    return binary_labels


def write_df_to_json_array_record(df, path):
    with array_record.ArrayRecordWriter(path, "w") as writer:
        for record in df.to_dicts():
            record_bytes = json.dumps(record).encode("utf-8")
            writer.write(record_bytes)


def get_datasources(percentile: int):
    proteinea_splits = {
        "train": "solubility_training.csv",
        "validation": "solubility_validation.csv",
        "test": "solubility_testing.csv",
    }

    plmsol_base_path = "experiments/plmsol_dataset"
    plmsol_splits = {
        "train": "train_dataset.fasta",
        "validation": "validation_dataset.fasta",
        "test": "test_dataset.fasta",
    }

    proteinea_train_df = pl.read_csv(
        "hf://datasets/proteinea/solubility/" + proteinea_splits["train"]
    )
    proteinea_test_df = pl.read_csv(
        "hf://datasets/proteinea/solubility/" + proteinea_splits["test"]
    )
    proteinea_validation_df = pl.read_csv(
        "hf://datasets/proteinea/solubility/" + proteinea_splits["validation"]
    )

    plmsol_dfs = {}
    for split_name, file_name in plmsol_splits.items():
        file_path = os.path.join(plmsol_base_path, file_name)
        seqs, string_labels = parse_plmsol_fasta(file_path)
        if seqs:
            binary_labels = convert_plmsol_labels_to_binary(string_labels)
            plmsol_dfs[split_name] = pl.DataFrame(
                {"sequences": seqs, "labels": binary_labels}
            )
            print(f"Loaded {len(seqs)} {split_name} samples from PLMSol.")
        else:
            plmsol_dfs[split_name] = pl.DataFrame(
                {"sequences": [], "labels": []},
                schema={"sequences": pl.String, "labels": pl.Int64},
            )

    plmsol_train_df = plmsol_dfs.get("train")
    plmsol_test_df = plmsol_dfs.get("test")
    plmsol_validation_df = plmsol_dfs.get("validation")

    assert plmsol_train_df is not None
    assert plmsol_test_df is not None
    assert plmsol_validation_df is not None

    train_df = pl.concat([proteinea_train_df, plmsol_train_df])
    test_df = pl.concat([proteinea_test_df, plmsol_test_df])
    validation_df = pl.concat([proteinea_validation_df, plmsol_validation_df])

    print(f"Total training samples: {len(train_df)}")
    print(f"Total test samples: {len(test_df)}")
    print(f"Total validation samples: {len(validation_df)}")

    print("\n--- Finalizing Data Preparation ---")
    train_df = train_df.with_columns(
        pl.col("sequences").str.len_chars().alias("length"),
    )
    max_protein_length = int(
        train_df.select(pl.col("length").quantile(percentile / 100)).item()
    )
    print(f"Max protein length set to: {max_protein_length}")

    train_source = DataFrameDataSource(train_df)
    test_source = DataFrameDataSource(test_df)
    validation_source = DataFrameDataSource(validation_df)

    print("\n--- Label Distribution ---")
    print("Train DF:\n", train_df.get_column("labels").value_counts())
    print("Test DF:\n", test_df.get_column("labels").value_counts())
    print("Validation DF:\n", validation_df.get_column("labels").value_counts())

    print("\n--- Sequence Length Statistics ---")
    print("Train DF:\n", train_df.describe())
    print("Test DF:\n", test_df.describe())
    print("Validation DF:\n", validation_df.describe())

    return train_source, test_source, validation_source, max_protein_length


def get_dataloaders(
    train_source,
    test_source,
    validation_source,
    epoch: int,
    batch_size: int,
    max_protein_length: int,
):
    train_index_sampler = grain.samplers.IndexSampler(
        num_records=len(train_source),
        shuffle=True,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=4 + epoch,
        num_epochs=1,
    )
    train_data_loader = grain.DataLoader(
        data_source=train_source,
        operations=[
            ProteinPreprocessor(max_protein_length),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=train_index_sampler,
    )

    test_dataset_size = len(test_source)
    validation_dataset_size = len(validation_source)
    test_index_sampler = grain.samplers.IndexSampler(
        num_records=test_dataset_size,
        shuffle=False,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=42,
        num_epochs=1,
    )
    test_data_loader = grain.DataLoader(
        data_source=test_source,
        operations=[
            ProteinPreprocessor(max_protein_length),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=test_index_sampler,
    )
    validation_index_sampler = grain.samplers.IndexSampler(
        num_records=validation_dataset_size,
        shuffle=False,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=42,
        num_epochs=1,
    )
    validation_data_loader = grain.DataLoader(
        data_source=validation_source,
        operations=[
            ProteinPreprocessor(max_protein_length),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=validation_index_sampler,
    )

    return train_data_loader, test_data_loader, validation_data_loader
