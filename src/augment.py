import math
import re
import random
import numpy as np
from typing import Union
from transformers import PreTrainedTokenizer

from src.datasets import ModelDataset, ModelSample, LabelPosition, ModelDatasetCompiler, TextSample


def generate_raw_positive_samples(
        dataset: ModelDataset,
        tokenizer: PreTrainedTokenizer
    ):
    samples = dataset.positive_dataset.fetch_all_samples()
    model_samples = [
        ModelSample(
            text=sample.text, 
            labels=[sample.label],
            positive=True,
            positions=[LabelPosition(label=sample.label, start_index=0, end_index=len(tokenizer.tokenize(sample.text)))]
        ) 
        for sample in samples
    ]
    return model_samples

def generate_raw_negative_samples(
        dataset: ModelDataset, 
        tokenizer: PreTrainedTokenizer,
        n: int, 
        max_length: int = 512
    ):
    samples = dataset.negative_dataset.sample(n)
    model_samples = []
    for sample in samples:
        text = sample.text
        tokenized_text = tokenizer.tokenize(sample.text)
        if len(tokenized_text) > max_length:
            tokenized_text = tokenized_text[:max_length]
            text = tokenizer.convert_tokens_to_string(tokenized_text)
        model_samples.append(
            ModelSample(
                text=text, 
                labels=[],
                positive=False
            )
        )
    return model_samples

def _get_positives(
        dataset: ModelDataset, 
        tokenizer: PreTrainedTokenizer,
        classes: list[str], 
        class_weights: dict[str, float] = dict(),
        max_labels: int = 2,
        class_exclusivity: bool = True,
)->tuple[list[TextSample], int, list[int]]:
    n_positive = np.random.randint(1, max_labels+1)
    positive_samples = dataset.positive_dataset.sample(
        n=n_positive, 
        classes=classes, 
        class_weights=class_weights,
        class_exclusivity=class_exclusivity,
    )
    positive_samples_length = sum([len(tokenizer.tokenize(sample.text)) for sample in positive_samples])
    positive_sample_lengths = [len(tokenizer.tokenize(sample.text)) for sample in positive_samples]
    return positive_samples, positive_samples_length, positive_sample_lengths

def _get_negatives(
        dataset: ModelDataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        split_on: list[str],
        bleed_allowance: float,
        min_positive_length: int
)->tuple[list[str], list[int]]:
    negative_sample = dataset.negative_dataset.sample(
        n=1
    )[0]
    pattern = "|".join(f"(?<={re.escape(x)})" for x in split_on)
    negative_segments = [x for x in re.split(pattern, negative_sample.text) if len(x.strip()) > 0]

    if not any([so.strip() in negative_segments[-1][-len(so)] for so in split_on]):
        negative_segments[-1] += np.random.choice(split_on)

    negative_segment_lengths = [len(tokenizer.tokenize(segment)) for segment in negative_segments]

    for i in range(len(negative_segments)):
        if sum(negative_segment_lengths[:i+1]) > max_length + bleed_allowance * min_positive_length:
            negative_segments = negative_segments[:i]
            break
    return negative_segments, negative_segment_lengths


def _consolidate_samples(
        negative_segments: list[str],
        positive_samples: list[TextSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        bleed_allowance: float,
        positive_sample_lengths: list[int]
)->tuple[str, list[LabelPosition], bool]:
    clip_check = False
    awaiting: list[Union[str, TextSample]] = negative_segments + positive_samples
    random.shuffle(awaiting)
    text = ""
    positions: list[LabelPosition] = []
    for i, sample in enumerate(awaiting):
        if isinstance(sample, str):
            if len(tokenizer.tokenize(text)) + len(tokenizer.tokenize(sample)) > max_length+bleed_allowance*min(positive_sample_lengths):
                break
            text += sample
        else:
            sample_text = sample.text
            sample_label = sample.label
            start_index = len(tokenizer.tokenize(text))
            end_index = start_index + len(tokenizer.tokenize(sample_text))
            if end_index > max_length + bleed_allowance * min(positive_sample_lengths):
                break
            text += sample_text
            positions.append(
                LabelPosition(
                    label=sample_label,
                    start_index=start_index,
                    end_index=end_index
                )
            )
            if i == 0:
                clip_check = True
            if i == len(awaiting)-1:
                clip_check = True
    return text, positions, clip_check

def _clipping(
        text: str,
        positions: list[LabelPosition],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        clip_check: bool
)->tuple[str, list[LabelPosition], bool]:
    sample_text_tokenized = tokenizer.tokenize(text)
    clipped_flag = False
    if len(sample_text_tokenized) > max_length:
        overage = len(sample_text_tokenized) - max_length
        clipped = sample_text_tokenized[math.ceil(overage/2):-math.ceil(overage/2)]
        assert len(clipped) <= max_length, f"Clipped text is longer than max_length: {len(clipped)} > {max_length}"
        text = tokenizer.convert_tokens_to_string(clipped)
        clipped_flag = True if clip_check else False
        for position in positions:
            position.start_index = max([position.start_index - math.ceil(overage/2), 0])
            position.end_index = min([position.end_index - math.ceil(overage/2), max_length])
    return text, positions, clipped_flag


def generate_augmented_sample(
        dataset: ModelDataset, 
        tokenizer: PreTrainedTokenizer,
        classes: list[str], 
        class_weights: dict[str, float] = dict(),
        max_length: int = 512,
        split_on: list[str] = [". ", "! ", "? ", "\n"],
        bleed_allowance: float = 0.0,
        max_labels: int = 2,
        class_exclusivity: bool = True,
):
    # Check if max_labels is greater than the number of classes
    if class_exclusivity:
        if max_labels > len(classes):
            raise ValueError(
                f"Cannot sample {max_labels} samples, only {len(classes)} classes available and class_exclusivity is set to True"
            )
    
    # Sample positive samples
    positive_samples, positive_samples_length, positive_sample_lengths = _get_positives(
        dataset=dataset, 
        tokenizer=tokenizer,
        classes=classes, 
        class_weights=class_weights,
        max_labels=max_labels,
        class_exclusivity=class_exclusivity,
    )
    
    # Sample negative samples
    if positive_samples_length < max_length:

        negative_segments, negative_segment_lengths = _get_negatives(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            split_on=split_on,
            bleed_allowance=bleed_allowance,
            min_positive_length=positive_samples_length
        )

        while True:
            text, positions, clip_check = _consolidate_samples(
                negative_segments=negative_segments,
                positive_samples=positive_samples,
                tokenizer=tokenizer,
                max_length=max_length,
                bleed_allowance=bleed_allowance,
                positive_sample_lengths=positive_sample_lengths
            )
            if len(positions) > 0:
                break
        
    else:
        while True:
            text, positions, clip_check = _consolidate_samples(
                negative_segments=[],
                positive_samples=positive_samples,
                tokenizer=tokenizer,
                max_length=max_length,
                bleed_allowance=bleed_allowance,
                positive_sample_lengths=positive_sample_lengths
            )
            if len(positions) > 0:
                break
    
    # Clip the text if it is too long
    text, positions, clipped_flag = _clipping(
        text=text,
        positions=positions,
        tokenizer=tokenizer,
        max_length=max_length,
        clip_check=clip_check
    )

    return ModelSample(
        text=text,
        labels=[x.label for x in positions],
        positive=True,
        positions=positions,
        clipped=clipped_flag
    )
        

def generate_dataset(
        dataset: ModelDataset,
        tokenizer: PreTrainedTokenizer,
        n_positive: int,
        classes: list[str],
        class_weights: dict[str, float] = dict(),
        max_length: int = 512,
        split_on: list[str] = [". ", "! ", "? ", "\n"],
        bleed_allowance: float = 0.0,
        max_labels: int = 2,
        class_exclusivity: bool = True,
)->ModelDatasetCompiler:
    model_samples = []
    model_samples += generate_raw_positive_samples(dataset=dataset, tokenizer=tokenizer)
    for _ in range(n_positive):
        model_samples.append(
            generate_augmented_sample(
                dataset=dataset,
                tokenizer=tokenizer,
                classes=classes,
                class_weights=class_weights,
                max_length=max_length,
                split_on=split_on,
                bleed_allowance=bleed_allowance,
                max_labels=max_labels,
                class_exclusivity=class_exclusivity
            )
        )
    model_samples += generate_raw_negative_samples(
        dataset=dataset,
        tokenizer=tokenizer,
        n=len(model_samples),
        max_length=max_length
    )
    return ModelDatasetCompiler(dataset=model_samples)