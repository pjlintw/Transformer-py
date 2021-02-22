# Transformer-py

The repository works on fine-tuning Huggingface's Transformer-based models for POS tagging. To reproduce the results,
follow the steps bellow.

## Installation

### Python version

* Python >= 3.8

### Environment

Create environment from file and activate the environment.

```
conda env create -f environment.yaml
conda activate fabian-pinjie
```

If conda fails to create environment from `environment.yaml`. This may be caused by the platform-specific build constraints in the file. Try create one by installing the important packages mannually. The `environment.yaml` was built in macOS.

**Note**: Running `conda env export > environment.yaml` will include all the 
dependencies conda automatically installed for you. Some dependecies may not work in different platforms.
We suggest you to use `--from-history` flag to export the packages to the enviroment setting file.
Make sure `conda` only exports the packages that you've explicitly asked for.

```
conda env export > enviroment.yaml --from-history
```

## Dataset and Preprocessing

### Dataset concatenation

We use `chtb_0223.gold_conll`, `phoenix_0001.gold_conll`, `pri_0016.gold_conll` and `wsj_1681.gold_conll` as the data for fine-tuning the pre-trained model.
These files are in the `data` folder. We combine them as one file `sample.conll` for preprocessing in next step.

```
cd data
cat *.gold_conll >> sample.conll
```

### Preprocessing

The file `sample.conll` contains irrelevant informations for training the neural nets.
We only need the sequence of observation, POS tags and the word position for the positional embedding in transformer. Running `data_preprocess.py` to extract `word position`, `word` and `POS tag` and write it to
`sample.tsv` in which `word position`, `word` and `POS tag` are separated by tab. 

The arguments `--dataset_name` and `output_dir` are the file to be passed to the program and the repository for the output file respectively. 

```python
python data_preprocess.py \
  --dataset_name sample.conll \
  --output_dir ./
```

Or just run the bash script `source ./run_preprocess.sh` in the command line. The output file `sample.tsv` will under the 
path `--output_dir`. 


### Data information

To get the information regarding the observations and POS taggings. Execute the script `data_information.py` to compute the
maximum, minumum and mean of the sequence length, number of examples, POS tags and its percentage.

The arguments `--dataset_name` and `output_dir` are the file to be passed to the program and the repository for the output file respectively. 

```python
python data_information.py \
  --dataset_name sample.tsv \
  --output_dir ./
```

or run `source ./run_information.sh` in the command line. The output file `sample.info` will be exported in the  `--output_dir` directory.
