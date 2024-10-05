# Text Classification Augmentor (WIP)
This repo is for augmenting text datasets for the purposes of sequence classification and span prediction tasks. It has been created for a specific project, however the use-case is likely applicable across other projects of a similar nature. The repo allows the creation of synthetic examples to increase sample size of text sequences, by permuting the text of importance within random segments of text.

## Original Use-Case
Classification and extraction of specific clauses within contract text data.

Within this there are three distinct tasks that this data augmentation repo can produce datasets for:
- Important / Not Important Text Chunk (Binary Classification)
- Text Chunk Classification (Multi-Label Classification)
- Text Extraction (via Span Prediction / Extractive Q&A)

For example, the first task can filter text chunks on whether they may contain clauses of interest, the 2nd task will classify which clauses are in the text chunk, and the final task can extract just the relevant clause text (sequentially over each clause present in the text).

# How To Use
1. Install the dependencies
```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```
2. Prepare the positive and negative sample datasets
```
The positive dataset should be a csv with a "text" column, and a "label" column. The text should ONLY contain specific text related to one of the classes you are seeking to predict, with any irrelevant text pruned.

The negative dataset should be a csv with a "text" column, containing text chunks of irrelevant text (i.e. does not include text that is relevant to positive samples).

Text chunk sizes (number of tokens) should be within the constraints of your chosen model's context window (e.g. <= 512 for BERT).

It is assumed positive samples are << 512 tokens.
```
3. Use the dataset factory to prepare the data (src.datasets.ModelDataset):
```python
from src.factory import dataset_factory

dataset = dataset_factory(
    positive_df=positive_df, # dataframe of positive examples
    negative_df=negative_df, # dataframe of negative examples
)
```
4. Generate the samples for the datasets:
```python
from src.augment import generate_dataset

...

dataset_generator = generate_dataset(
    dataset=dataset, # the dataset produced via dataset_factory
    tokenizer=tokenizer, # your chosen tokenizer for the model
    n_positive=1000, # number of positive samples to augment
    classes=..., # list of class names that are contained in the positive_df labels
    max_length=512, # max token length
    bleed_allowance=0.2, # allows partially clipped positive text in a generated sample (set to zero to disable)
    max_labels=5, # number of positive labels per generated sample
)
```
5. Access the generated datasets:
```python
...

binary_cls_dataset = dataset_generator.binary_importance_dataset()
multi_label_cls_dataset = dataset_generator.multi_label_classification_dataset()
span_prediction_dataset = dataset_generator.span_prediction_dataset()
```

# Notes
- It is advised that you add any unknown tokens to the tokenizer you use prior to generating the dataset, else you will end up with some text being slightly garbled due to [UNK] tokens produced.
- This repo is a work in progress, there are very likely bugs in the code and it is yet to be tested.