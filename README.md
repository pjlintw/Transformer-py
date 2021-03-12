# Transformer-py: a Flexible Framework for POS tagging.

[**Data**](#dataset-and-preprocessing) | [**Training**](#run-bert-variants-for-pos-tagging)

The repository works on fine-tuning of the pre-trained Transformer-based models for Parts-of-speech (POS) tagging. We leverage `chtb_0223.gold_conll`, `phoenix_0001.gold_conll`, `pri_0016.gold_conll` and `wsj_1681.gold_conll` annotated file as dataset for fine-tuning. To reproduce the results, follow the steps bellow.

In the literature, the intial layers is used to encode general, semantic-irrelevant information. The middle layers usually enabels to produce information-rich represenations. The latter layers are good at encoding the abstractive and task-orientic semantic representation. We develop a flixible framework to run such experiements. 

* New Februray 22th, 2021: Data preprocessing and data information.
* New March 8th, 2021: Train the BERT and custom model, dataset loading script.
    
## TO-DO
* Experiements of Linear Probing.
* Experiements of data efficiency.


## Installation

### Python version

* Python >= 3.8

### Environment

Create environment from file and activate the environment.

```
conda env create -f environment.yaml
conda activate fabian-pinjie
```

If conda fails to create environment from `environment.yaml`. This may be caused by the platform-specific build constraints in the file. Try create one by installing the important packages manually. The `environment.yaml` was built in macOS.

**Note**: Running `conda env export > environment.yaml` will include all the 
dependencies conda automatically installed for you. Some dependecies may not work in different platforms.
We suggest you to use `--from-history` flag to export the packages to the enviroment setting file.
Make sure `conda` only exports the packages that you've explicitly asked for.

```
conda env export > environment.yaml --from-history
```

## Dataset and Preprocessing

### Dataset concatenation

We use `chtb_0223.gold_conll`, `phoenix_0001.gold_conll`, `pri_0016.gold_conll` and `wsj_1681.gold_conll` as the data for fine-tuning the pre-trained model.
These files are in the `data` folder. We combine them as one file `sample.conll` for preprocessing in next step.

```
cd data
cat *.gold_conll >> sample.conll
```

In the following steps, you will preprcesse the collected file `sample.conll`, then split them into `sample.train`, `sample.dev` and `sample.test`
for building the datasets. You have to change the relative path to `--datasets_name` if you're using different file directory .


### Preprocessing and Dataset Splitting

The file `sample.conll` contains irrelevant informations for training the neural nets.
We only need the sequence of observation, POS tags and the word position for the positional embedding in transformer. Running `data_preprocess.py` to extract `word position`, `word` and `POS tag` and write it to
`sample.tsv` in which `word position`, `word` and `POS tag` are separated by tab. 

The arguments `--dataset_name` and `output_dir` are the file to be passed to the program and the repository for the output file respectively. 

It generates `sample.tsv` for all examples and `sample.train`, `sample.dev` and `sample.test` for the network training.  The examples will be suffled in the scripts and split into `train`, `validation` and `test` files.  The arugements `--eval_samples` and `--test_samples`
decide the number of samples will be selected from examples. In OntoNotes datasets, we select 67880 for training set, 2000 for validation and test sets respectively. To preprocesse and split the datasets, you need to run the code bellow. 

```python
python data_preprocess.py \
  --dataset_name sample.conll \
  --output_dir ./ \
  --eval_samples 2000 \
  --test_samples 2000 
```

Or just run the bash script `source ./run_preprocess.sh` in the command line. The output file `sample.tsv` will under the 
path `--output_dir`. You will get the result.

```
Loading 69880 examples
Seed 49 is used to shuffle examples
Saving 69880 examples to sample.tsv
Saving 65880 examples to sample.train
Saving 2000 examples to sample.dev
Saving 2000 examples to sample.test
```

Make sure that **the datasets** you passed to the argument `--dataset_name` has larger number examples for splitting out develop and test set. The examples files may have no example, if the splitting number for eval and test sets is greater than the example in `sample.conll`.


### Data Information

To get the information regarding the observations and POS taggings. Execute the script `data_information.py` to compute 
the percentiles, maximum, minumum and mean of the sequence length, number of examples, POS tags and its percentage.

The arguments `--dataset_name` and `output_dir` are the file to be passed to the program and the repository for the output file respectively. 

```python
python data_information.py \
  --dataset_name sample.tsv \
  --output_dir ./
```

or run `source ./run_information.sh` in the command line. The output file `sample.info` will be exported in the  `--output_dir` directory.

### Train with Custom OntoNotes v4.0

We use our dataset loading script `ontonotes_v4.py`for creating dataset. The script builds the train, validation and test sets from those 
dataset splits obtained by the `data_preprocess.py` program. Make sure the dataset split files `sample.train`, `sample.dev` , and `sample.test` are included in the datasets folder `/ontonotes-4.0/` your dataset folder. 

## Save the Results

We sugguest that using `Weights & Biass` to save the configuration, loss and evaluation metrics for you.
To connect your own `Weights & Biass` account. Just installing the packages using `pip install wandb`
and login it. The `trainer` in `run_pos.py` will automaticaaly log the `TrainingArguments`, losses and evalaution
metircs and model information to your account. 

```
wandb login
```

You can specify which project folder for saving files. For exmaple, set project name 
to the environment variable.

```
export WANDB_PROJECT=TEST_PROJECT
export WANDB_WATCH=all
```

## Run BERT variants for POS tagging

We evaluate the BERT on linear probing test to see which layer capture more linsutic structure 
information in therir contextual representaitons. The output layers for classifing the POS tags are added on the different layers of BERT. We only train these layer's weights.

We treats BERT as a feature extractor to provide fixed pre-trained contextual embeddings.
In the script, we set `requires_grad` false for BERT model. If you would like to fine-tune the whole the model, just comment those lines.

In certain cases, rather than fine-tuning the entire pre-trained model end-to-end, it can be beneficial to obtained pre-trained contextual embeddings, which are fixed contextual representations of each input token generated from the hidden layers of the pre-trained model. This should also mitigate most of the out-of-memory issues.

We found that executing time of  64 minibatch size trained with maximal sequence length is pretty slow. 
The maximal sequence length, in OntoNotoes is 228, is usually an extrem case. We gains huge improvement on the runtime for an minibatch by using  using 63 to `max_seq_length` covering 99% of sequence length.


### Train the offficail BERT model

The official Huggingface BERT for sequence labeling task using `BertForTokenClassification` class. 
The model levreages a pre-trained BERT, droput and an classifier layer.
To run these settings, you can run

```python
python run_pos.py \
 --model_name_or_path bert-base-cased \
 --output_dir /tmp/pos-exp-1 \
 --task_name pos \
 --dataset_script ontonotes_v4.py \
 --max_seq_length 63 \
 --per_device_train_batch_size 48 \
 --per_device_eval_batch_size 48 \
 --num_train_epochs 3 \
 --do_train \
 --do_eval \
 --do_predict
 --learning_rate 1e-2 
```

### Train Linear Probing BERT  

Linear probing BERT is an architecture to extract the fixed contextual representations from the BERT.
It aims to evalaute which layer captures linguistic structure information in their features.

The custom model based `bert-base-cased`. Therefore, it has one embedding layer in 12 BERT layer in 
BERT model. If you use a classifier on top of 12th BERT's layer. It is same as the standard BERT that 
`BertForTokenClassification` class creats for you. 

To train BERT model on linear probing setting, you have to specify `linear-probing-bert.py` to
the option `--model_name_or_path` and pass integer indicating on which BERT's layer the classifier heads on.

```python
 python run_pos.py \
 --model_name_or_path models/linear-probing-bert.py \
 --output_dir /tmp/pos-exp-1 \
 --task_name pos \
 --dataset_script ontonotes_v4.py \
 --max_seq_length 63 \
 --per_device_train_batch_size 48 \
 --per_device_eval_batch_size 48 \
 --max_steps 120 \
 --do_train \
 --do_eval \
 --do_predict \
 --max_train_samples 10000 \
 --max_val_samples 300 \
 --max_test_samples 300 \
 --logging_first_step \
 --logging_steps 5 \
 --learning_rate 1e-2 \
 --evaluation_strategy steps \
 --eval_steps 10 \
 --to_layer 2 
```


### Train your custom model

Run the costom model via the path `models/custom-model-demo.py`.
You can define your custom model by modifing the demo script.

```python
python run_pos.py \
 --model_name_or_path models/custom-model-demo.py \
 --output_dir /tmp/pos-exp-1 \
 --task_name pos \
 --dataset_script ontonotes_v4.py \
 --max_seq_length 63 \
 --per_device_train_batch_size 16 \
 --per_device_eval_batch_size 8 \
 --num_train_epochs 3 \
 --do_train \
 --do_eval \
 --do_predict
  --learning_rate 1e-2 \
```

### Quicker training  

If you'd like to furthe develop the model or dedugging it. 
Th options `max_train_samples`, `max_vall_samples` and `max_test_samples` allow you to truncate the number of examples.
They recieve digits digit format.

```
python run_pos.py \
 --model_name_or_path models/custom-model-demo.py \
 --output_dir /tmp/pos-exp-1 \
 --task_name pos \
 --dataset_script ontonotes_v4.py \
 --max_seq_length 63 \
 --per_device_train_batch_size 24 \
 --per_device_eval_batch_size 8 \
 --num_train_epochs 3 \
 --do_train \
 --do_eval \
 --do_predict \
 --max_train_samples 10000 \
 --max_val_samples 1000 \
 --max_test_samples 1000 \
 --logging_steps 20 
 --learning_rate 1e-2 \
```
