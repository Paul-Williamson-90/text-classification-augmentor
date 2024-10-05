from src.datasets import PositiveDataset, NegativeDataset, ModelDataset
import pandas as pd
from typing import Optional


def dataset_factory_from_save(
    directory: str,
) -> ModelDataset:
    return ModelDataset.load_dataset(directory)


def dataset_factory_from_df(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> ModelDataset:
    positive_dataset = PositiveDataset.from_df(positive_df, text_col=text_col, label_col=label_col)
    negative_dataset = NegativeDataset.from_df(negative_df, text_col=text_col)
    return ModelDataset(
        positive_dataset=positive_dataset, negative_dataset=negative_dataset
    )


def dataset_factory(
    directory: Optional[str] = None,
    positive_df: Optional[pd.DataFrame] = None,
    negative_df: Optional[pd.DataFrame] = None,
    text_col: str = "text",
    label_col: str = "label",
) -> ModelDataset:
    if directory and (isinstance(positive_df, pd.DataFrame) or isinstance(negative_df, pd.DataFrame)):
        raise ValueError("Cannot provide both a directory and dataframes")
    if directory:
        return dataset_factory_from_save(directory)
    elif isinstance(positive_df, pd.DataFrame) and isinstance(negative_df, pd.DataFrame):
        return dataset_factory_from_df(positive_df, negative_df, text_col, label_col)
    else:
        raise ValueError(
            "Must provide either a directory or positive and negative dataframes"
        )
