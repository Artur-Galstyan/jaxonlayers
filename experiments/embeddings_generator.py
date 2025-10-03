import hashlib
import os

import grain
import jax.numpy as jnp
import numpy as np
import polars as pl
import tensorflow_datasets as tfds
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

PERCENTILE = 90

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
vocab = ["Z"] + list(amino_acids)
int_to_char = {i: char for i, char in enumerate(vocab)}
char_to_int = {char: i for i, char in enumerate(vocab)}

splits = {
    "train": "solubility_training.csv",
    "validation": "solubility_validation.csv",
    "test": "solubility_testing.csv",
}
train_df = pl.read_csv("hf://datasets/proteinea/solubility/" + splits["train"])
test_df = pl.read_csv("hf://datasets/proteinea/solubility/" + splits["test"])
validation_df = pl.read_csv(
    "hf://datasets/proteinea/solubility/" + splits["validation"]
)

train_df = train_df.with_columns(
    pl.col("sequences").str.len_chars().alias("length"),
)
max_protein_length = int(
    train_df.select(pl.col("length").quantile(PERCENTILE / 100)).item()
)


class ProteinPreprocessor(grain.transforms.Map):
    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        sequence = sequence[:max_protein_length]
        sequence = sequence.ljust(max_protein_length, "Z")
        indices = [char_to_int[aa] for aa in sequence]
        return {
            "features": jnp.array(indices, jnp.int32),
            "sequence": sequence,
            "label": jnp.zeros(shape=(2,)).at[label].set(1),
        }


batch_size = 2048
builder = tfds.dataset_builders.CroissantBuilder(
    jsonld="https://huggingface.co/api/datasets/proteinea/solubility/croissant",
    file_format="array_record",
)
builder.download_and_prepare()

train_source, test_source = builder.as_data_source(split=["train", "test"])

dataset_size = len(train_source)
print(f"Dataset size: {dataset_size}")

train_index_sampler = grain.samplers.IndexSampler(
    num_records=dataset_size,
    shard_options=grain.sharding.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=False
    ),
    num_epochs=1,
)

train_data_loader = grain.DataLoader(
    data_source=train_source,  # pyright: ignore
    operations=[ProteinPreprocessor(), grain.transforms.Batch(batch_size=batch_size)],
    sampler=train_index_sampler,
)

dataset_size = len(test_source)
print(f"Dataset size: {dataset_size}")

test_index_sampler = grain.samplers.IndexSampler(
    num_records=dataset_size,
    shard_options=grain.sharding.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=False
    ),
    num_epochs=1,
)

test_data_loader = grain.DataLoader(
    data_source=test_source,  # pyright: ignore
    operations=[ProteinPreprocessor(), grain.transforms.Batch(batch_size=batch_size)],
    sampler=test_index_sampler,
)


client = ESMC.from_pretrained("esmc_300m").to("cpu")  # or "cpu"

# for data in tqdm(train_data_loader, total=dataset_size // batch_size):
#     xs = []
#     for seq in data["sequence"]:
#         seq_sha = hashlib.sha256(seq.encode("utf-8")).hexdigest()
#         if not os.path.exists(f"experiments/embeddings/{seq_sha}.npy"):
#             protein = ESMProtein(sequence=seq)
#             protein_tensor = client.encode(protein)
#             logits_output = client.logits(
#                 protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
#             )
#             assert logits_output.embeddings is not None
#             embedding = logits_output.embeddings.squeeze(1)
#             embedding = embedding.cpu().numpy()
#             np.save(f"experiments/embeddings/{seq_sha}", embedding)

for data in tqdm(test_data_loader, total=dataset_size // batch_size):
    xs = []
    for seq in data["sequence"]:
        seq_sha = hashlib.sha256(seq.encode("utf-8")).hexdigest()
        if not os.path.exists(f"experiments/embeddings/{seq_sha}.npy"):
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            assert logits_output.embeddings is not None
            embedding = logits_output.embeddings.squeeze(1)
            embedding = embedding.cpu().numpy()
            np.save(f"experiments/embeddings/{seq_sha}", embedding)
