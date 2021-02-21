# Transformer-py

Fine-tuning Huggingface's Transformer-based models for POS tagging.

## Installation

* Python >= 3.8

## Dataset and Preprocessing

### Dataset concatenation


```
cat *.gold_conll >> 
```

### Preprocessing


```python
python data_preprocess.py \
  --dataset_name sample.conll \
  --output_dir ./
```

or run `source ./run_preprocess.sh` in the command line.


### Data information

To get the information regarding the POS tagging. Execute the script 
`run_preprocess.py`.

The Arguemts are....

```python
python data_information.py \
  --dataset_name sample.tsv \
  --output_dir ./
```
or run `source ./run_information.sh` in the command line.
