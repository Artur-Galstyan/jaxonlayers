import json
import os

import equinox as eqx
import grain
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable, cast
from grain.samplers import IndexSampler
from grain.sources import ArrayRecordDataSource
from grain.transforms import Batch, Map
from jaxtyping import Array, Int
from pydantic import BaseModel


class ARCData(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    input: np.ndarray
    output: np.ndarray


class ARC(BaseModel):
    train: list[ARCData]
    test: list[ARCData]


def get_arcs(arcs_dir):
    arcs: list[ARC] = []
    for file in os.listdir(arcs_dir):
        with open(f"{arcs_dir}/{file}", "rb") as f:
            data = json.load(f)
            for item in data["train"]:
                item["input"] = np.array(item["input"])
                item["output"] = np.array(item["output"])
            for item in data["test"]:
                item["input"] = np.array(item["input"])
                item["output"] = np.array(item["output"])
            data = ARC.model_validate(data)
            arcs.append(data)

    return arcs


def get_size_infos(*arcs: list[ARC]) -> tuple[int, int, int]:
    max_size = 0
    max_train = 0
    max_test = 0
    for arc_list in arcs:
        for arc in arc_list:
            max_train = len(arc.train) if len(arc.train) > max_train else max_train
            max_test = len(arc.test) if len(arc.test) > max_test else max_test
            for el in arc.train + arc.test:
                max_size = len(el.input) if len(el.input) > max_size else max_size
    return max_size, max_train, max_test


arc_train_dir = "experiments/arc-1-data/training"
arc_eval_dir = "experiments/arc-1-data/evaluation"
train_arcs: list[ARC] = get_arcs(arc_train_dir)
eval_arcs: list[ARC] = get_arcs(arc_eval_dir)

max_size, max_train, max_test = get_size_infos(train_arcs, eval_arcs)


class DataSource(ArrayRecordDataSource):
    def __init__(self, data_list: list[ARC]):
        self._data = data_list

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, record_key: int) -> ARC:
        return self._data[record_key]


class Preprocessor(Map):
    def __init__(
        self, target_size: int, max_train: int, max_test: int, pad_value: int = -1
    ):
        self.target_size = target_size
        self.max_train = max_train
        self.max_test = max_test
        self.pad_value = pad_value

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        arr = np.array(grid)
        h, w = arr.shape
        padded = np.full(
            (self.target_size, self.target_size), self.pad_value, dtype=np.int32
        )
        padded[:h, :w] = arr
        return padded

    def _make_dummy(self) -> ARCData:
        dummy = np.full(
            (self.target_size, self.target_size), self.pad_value, dtype=np.int32
        )
        return ARCData(input=dummy, output=dummy)

    def map(self, element: ARC):
        train_inputs = []
        train_outputs = []
        for d in element.train:
            train_inputs.append(self._pad_grid(d.input))
            train_outputs.append(self._pad_grid(d.output))

        test_inputs = []
        test_outputs = []
        for d in element.test:
            test_inputs.append(self._pad_grid(d.input))
            test_outputs.append(self._pad_grid(d.output))

        train_mask = np.array(
            [1 if i < len(element.train) else 0 for i in range(self.max_train)]
        )
        test_mask = np.array(
            [1 if i < len(element.test) else 0 for i in range(self.max_test)]
        )

        dummy = np.full(
            (self.target_size, self.target_size), self.pad_value, dtype=np.int32
        )
        while len(train_inputs) < self.max_train:
            train_inputs.append(dummy)
            train_outputs.append(dummy)
        while len(test_inputs) < self.max_test:
            test_inputs.append(dummy)
            test_outputs.append(dummy)

        return (
            np.stack(train_inputs),
            np.stack(train_outputs),
            train_mask,
            np.stack(test_inputs),
            np.stack(test_outputs),
            test_mask,
        )


train_datasource = DataSource(train_arcs)
train_index_sampler = IndexSampler(num_records=len(train_datasource), num_epochs=5)
train_data_loader = grain.DataLoader(
    data_source=train_datasource,
    operations=[
        Preprocessor(target_size=max_size, max_train=max_train, max_test=max_test),
        Batch(batch_size=4),
    ],
    sampler=train_index_sampler,
    worker_count=0,
)

eval_datasource = DataSource(eval_arcs)
eval_index_sampler = IndexSampler(num_records=len(eval_datasource), num_epochs=5)
eval_data_loader = grain.DataLoader(
    data_source=eval_datasource,
    operations=[
        Preprocessor(target_size=max_size, max_train=max_train, max_test=max_test),
        Batch(batch_size=4),
    ],
    sampler=eval_index_sampler,
    worker_count=0,
)


def evaluate(model: Callable, eval_data_loader: grain.DataLoader):
    total_correct = 0
    total_count = 0
    batched_model = eqx.filter_vmap(model)

    for batch in eval_data_loader:
        (
            train_inputs,
            train_outputs,
            train_mask,
            test_inputs,
            test_outputs,
            test_mask,
        ) = batch

        predictions = batched_model(
            train_inputs, train_outputs, train_mask, test_inputs, test_mask
        )

        exact_match = (predictions == test_outputs).all(axis=(-1, -2))

        masked_correct = (exact_match * test_mask).sum()
        masked_count = test_mask.sum()

        total_correct += masked_correct
        total_count += masked_count

    accuracy = total_correct / total_count
    return accuracy


class Model(eqx.Module):
    def __init__(self):
        pass

    def __call__(
        self,
        train_input: Int[Array, "n_train_examples width height"],
        train_output: Int[Array, "n_train_examples width height"],
        train_mask: Int[Array, "n_train_examples"],
        test_input: Int[Array, "n_test_examples width height"],
        test_mask: Int[Array, "n_test_examples"],
    ):
        return test_input


model = Model()
acc = evaluate(
    model,
    eval_data_loader,
)
print(f"{acc=}")
