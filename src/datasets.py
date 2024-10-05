from uuid import uuid4, UUID
from typing import Union, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel


class TextSample(BaseModel):
    _id: UUID = uuid4()
    text: str
    label: str

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return self.text


class ClassSamples(BaseModel):
    class_name: str
    samples: list[TextSample]

    def __len__(self) -> int:
        return len(self.samples)

    def add_sample(self, text: str) -> None:
        self.samples.append(TextSample(text=text, label=self.class_name))

    def fetch_sample_ids(self) -> list[UUID]:
        return [sample._id for sample in self.samples]

    def fetch_sample_by_id(self, _id: Union[UUID, str]) -> Union[TextSample, None]:
        if isinstance(_id, str):
            _id = UUID(_id)
        for sample in self.samples:
            if sample._id == _id:
                return sample
        return None

    def sample(
        self, 
        n: int,
    ) -> list[TextSample]:
        samples = self.samples
        replace = True if n > len(samples) else False
        retrieved = np.random.choice(
            samples, n, replace=replace
        )
        if isinstance(retrieved, TextSample):
            return [retrieved]
        return retrieved

    def n_samples_not_in(self, exclude: list[UUID]) -> int:
        return len([sample for sample in self.samples if sample._id not in exclude])


class PositiveDataset:
    dataset: dict[str, ClassSamples] = dict()

    def add_sample(self, class_name: str, text: str):
        if class_name not in self.dataset:
            self.dataset[class_name] = ClassSamples(class_name=class_name, samples=[])
        self.dataset[class_name].samples.append(TextSample(text=text, label=class_name))

    def add_samples(self, class_name: str, texts: list[str]):
        for text in texts:
            self.add_sample(class_name=class_name, text=text)

    def fetch_sample_ids(self) -> list[UUID]:
        sample_ids = []
        for class_name in self.dataset:
            sample_ids.extend(self.dataset[class_name].fetch_sample_ids())
        return sample_ids

    def fetch_sample_by_id(self, _id: Union[UUID, str]) -> Union[TextSample, None]:
        if isinstance(_id, str):
            _id = UUID(_id)
        for class_name in self.dataset:
            sample = self.dataset[class_name].fetch_sample_by_id(_id)
            if sample is not None:
                return sample
        return None
    
    def fetch_all_samples(self) -> list[TextSample]:
        samples = []
        for class_name in self.dataset:
            samples.extend(self.dataset[class_name].samples)
        return samples

    def sample(
        self,
        n: int,
        classes: Optional[list[str]] = [],
        class_weights: Optional[dict[str, float]] = {},
        class_exclusivity: bool = True,
    ) -> list[TextSample]:
        if class_exclusivity:
            if n > len(classes):
                raise ValueError(
                    f"Cannot sample {n} samples without replacement, only {len(classes)} classes available"
                )

        if len(classes) == 0:
            classes = list(self.dataset.keys())
        else:
            classes = classes.copy()

        if len(class_weights) > 0:
            class_weights = self.prepare_class_weights(classes, class_weights)

        else:
            class_weights = {class_name: 1 / len(classes) for class_name in classes}

        samples: list[TextSample] = []
        
        while len(samples) < n:
            class_name = np.random.choice(
                classes, p=[class_weights[class_name] for class_name in classes]
            )
            samples.extend(
                self.dataset[class_name].sample(1)
            )
            if class_exclusivity:
                classes.remove(class_name)
                class_weights.pop(class_name)
                class_weights = self.prepare_class_weights(classes, class_weights)
        return samples
    
    def prepare_class_weights(
        self, classes: list[str], class_weights: dict[str, float]
    ) -> dict[str, float]:
        assert all(
            class_name in class_weights for class_name in classes
        ), "class_weights must have a weight for each class in classes"
        class_weights = class_weights.copy()
        class_weights = {
            class_name: class_weights[class_name] for class_name in classes
        }
        class_weights = {
            class_name: class_weights[class_name] / sum(class_weights.values())
            for class_name in class_weights
        }
        return class_weights

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, text_col: str, label_col: str
    ) -> "PositiveDataset":
        assert text_col in df.columns, f"{text_col} not in columns of df"
        assert label_col in df.columns, f"{label_col} not in columns of df"
        dataset = cls()
        for _, row in df.iterrows():
            assert row[label_col] != "negative", "label cannot be negative"
            dataset.add_sample(
                class_name=row[label_col], text=row[text_col]
            )
        return dataset

    def __len__(self) -> int:
        return sum([len(self.dataset[class_name]) for class_name in self.dataset])

    def save_dataset(self, directory: str) -> None:
        with open(directory + "/positive_dataset.tsv", "w") as f:
            for class_name in self.dataset:
                for sample in self.dataset[class_name].samples:
                    f.write(f"{sample.text}\t{sample.label}\n")

    @classmethod
    def load_dataset(cls, directory: str) -> "PositiveDataset":
        dataset = cls()
        with open(directory + "/positive_dataset.tsv", "r") as f:
            for line in f:
                text, label = line.strip().split("\t")
                dataset.add_sample(class_name=label, text=text)
        return dataset


class NegativeDataset:
    dataset: ClassSamples = ClassSamples(class_name="negative", samples=[])

    def sample(
        self,
        n: int,
    ) -> list[TextSample]:
        return self.dataset.sample(n)

    def __len__(self) -> int:
        return len(self.dataset)

    def save_dataset(self, directory: str) -> None:
        with open(directory + "/negative_dataset.tsv", "w") as f:
            for sample in self.dataset.samples:
                f.write(f"{sample.text}\t{sample.label}\n")

    @classmethod
    def load_dataset(cls, directory: str) -> "NegativeDataset":
        dataset = cls()
        with open(directory + "negative_dataset.tsv", "r") as f:
            for line in f:
                text, label = line.strip().split("\t")
                dataset.dataset.add_sample(text=text, label=label)
        return dataset

    @classmethod
    def from_df(cls, df: pd.DataFrame, text_col: str) -> "NegativeDataset":
        assert text_col in df.columns, f"{text_col} not in columns of df"
        dataset = cls()
        for _, row in df.iterrows():
            dataset.dataset.add_sample(text=row[text_col])
        return dataset
    
class ModelDataset:

    def __init__(
        self,
        positive_dataset: PositiveDataset,
        negative_dataset: NegativeDataset,
    ):
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset

    @classmethod
    def load_dataset(cls, directory: str) -> "ModelDataset":
        positive_dataset = PositiveDataset.load_dataset(directory)
        negative_dataset = NegativeDataset.load_dataset(directory)
        return cls(positive_dataset, negative_dataset)


class LabelPosition(BaseModel):
    label: str
    start_index: int
    end_index: int


class ModelSample(BaseModel):
    _id: UUID = uuid4()
    text: str
    labels: list[str]
    positive: bool
    positions: list[LabelPosition] = []
    clipped: bool = False


class ModelDatasetCompiler:

    def __init__(self, dataset: list[ModelSample]):
        self.dataset = dataset

    def binary_importance_dataset(self)->pd.DataFrame:
        data = {"text": [], "label": [], "clipped": []}
        for sample in self.dataset:
            data["text"].append(sample.text)
            data["label"].append(int(sample.positive))
            data["clipped"].append(int(sample.clipped))
        return pd.DataFrame(data)
    
    def multi_label_classification_dataset(self)->pd.DataFrame:
        samples = [x for x in self.dataset if x.positive]
        labels = list(set([y for x in samples for y in x.labels]))
        data = {"text": [], "clipped": []}
        for label in labels:
            data[label] = []
        for sample in samples:
            data["text"].append(sample.text)
            for label in labels:
                if label in sample.labels:
                    data[label].append(1)
                else:
                    data[label].append(0)
            data["clipped"].append(int(sample.clipped))
        return pd.DataFrame(data)
    
    def span_prediction_dataset(self)->pd.DataFrame:
        samples = [x for x in self.dataset if x.positive]
        data = {"text": [], "question": [], "start_index": [], "end_index": [], "clipped": []}
        
        for sample in samples:
            for position in sample.positions:
                data["text"].append(sample.text)
                data["question"].append(position.label)
                data["start_index"].append(position.start_index)
                data["end_index"].append(position.end_index)
                data["clipped"].append(int(sample.clipped))
        
        return pd.DataFrame(data)