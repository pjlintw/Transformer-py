# Transformer-py: a Flexible Framework for POS tagging.

[**Data**](#dataset-and-preprocessing) | [**Training**](#run-bert-variants-for-pos-tagging) | [**Linear-BERT**](#train-linear-bert)


The repository works on fine-tuning of the pre-trained Transformer-based models for Parts-of-speech (POS) tagging. We leverage `chtb_0223.gold_conll`, `phoenix_0001.gold_conll`, `pri_0016.gold_conll` and `wsj_1681.gold_conll` annotated file as dataset example for fine-tuning. To reproduce the results, follow the steps below.

In the literature, the initial layers are used to encode general, semantic-irrelevant information. The middle layers usually enable them to produce information-rich representations. The latter layers are good at encoding the abstractive and task-oriented semantic representation. We develop a flexible framework to run such experiments. 

* New February 22th, 2021: Data preprocessing and data information.
* New March 8th, 2021: Train the BERT and custom model, dataset loading script.
    


## Installation

### Python version

* Python >= 3.8

### Environment

Create an environment from file and activate the environment.

```
conda env create -f environment.yaml
conda activate fabian-pinjie
```

If conda fails to create an environment from `environment.yaml`. This may be caused by the platform-specific build constraints in the file. Try to create one by installing the important packages manually. The `environment.yaml` was built in macOS.

**Note**: Running `conda env export > environment.yaml` will include all the 
dependencies conda automatically installed for you. Some dependencies may not work in different platforms.
We suggest you to use the `--from-history` flag to export the packages to the environment setting file.
Make sure `conda` only exports the packages that you've explicitly asked for.

```
conda env export > environment.yaml --from-history
```

## Dataset and Preprocessing

### Dataset concatenation

We use `chtb_0223.gold_conll`, `phoenix_0001.gold_conll`, `pri_0016.gold_conll` and `wsj_1681.gold_conll` as the data for fine-tuning the pre-trained model.
These files are in the `data` folder. We combine them as one file `sample.conll` for preprocessing in the next step.

```
cd data
cat *.gold_conll >> sample.conll
```

In the following steps, you will preprocess the collected file `sample.conll`, then split them into `sample.train`, `sample.dev` and `sample.test`
for building the datasets. You have to change the relative path to `--datasets_name` if you're using a different file directory .


### Preprocessing and Dataset Splitting

The file `sample.conll` contains irrelevant information for training the neural nets.
We only need the sequence of observation, POS tags and the word position for the positional embedding in the transformer. Running `data_preprocess.py` to extract `word position`, `word` and `POS tag` and write it to
`sample.tsv` in which `word position`, `word` and `POS tag` are separated by tab. 

The arguments `--dataset_name` and `--output_dir` are the file to be passed to the program and the repository for the output file respectively. 

It generates `sample.tsv` for all examples and `sample.train`, `sample.dev` and `sample.test` for the network training.  The examples will be shuffled in the scripts and split into `train`, `validation` and `test` files.  The arguments `--eval_samples` and `--test_samples`
decide the number of samples will be selected from examples. In OntoNotes datasets, we select 67880 for training set, 2000 for validation and test sets respectively. To preprocess and split the datasets, you need to run the code below. 

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

Make sure that you pass the correct **datasets** to the `--dataset_name`argument and it has enough examples for splitting out develop and test set. The output files may have no example, if the numbers of eval and test exmaples are more than the examples in the `sample.conll`

### Data Information

To get the information regarding the observations and POS taggings. Execute the script `data_information.py` to compute 
the percentiles, maximum, minimum and mean of the sequence length, number of examples, POS tags and its percentage.

The arguments `--dataset_name` and `output_dir` are the file to be passed to the program and the repository for the output file respectively. 

```python
python data_information.py \
  --dataset_name sample.tsv \
  --output_dir ./
```

or run `source ./run_information.sh` in the command line. The output file `sample.info` will be exported in the  `--output_dir` directory.

### Train with Custom OntoNotes v4.0

We use our dataset loading script `ontonotes_v4.py`for creating dataset. The script builds the train, validation and test sets from those 
dataset splits obtained by the `data_preprocess.py` program. 
Make sure the dataset split files `sample.train`, `sample.dev` , and `sample.test` are included in the datasets folder `data/` your dataset folder.

If you get an error message like:

```
pyarrow.lib.ArrowTypeError: Could not convert 1 with type int: was not a sequence or recognized null for conversion to list type
```

You may have run other datasets in the same folder before. The Huggingface already created `.arrow` files once you run a loading script. These files are for reloading the datasets quickly.

Try to move the dataset you would like to use to the other folder and modify the path in the loading scipt. 

Or delete the relavent folder and files in the `.cache` for datasets. `cd ~/USERS_NAME/.cache/huggingface/datasets/` and `rm -r *`. This means that all the loading records will be removed and
 Hugginface will create the `.arrows` files again, including the previous laoding records. 


## Save the Results

We suggest that using `Weights & Biass` to save the configuration, loss and evaluation metrics for you.
To connect your own `Weights & Biass` account. Just installing the packages using `pip install wandb`
and login it. The `trainer` in `run_pos.py` will automatically log the `TrainingArguments`, losses and evaluation
metrics and model information to your account. 

```
wandb login
```

You can specify which project folder for these files. For example, set project name to the environment variable.

```
export WANDB_PROJECT=TEST_PROJECT
export WANDB_WATCH=all
```

## Run BERT variants for POS tagging

We evaluate the BERT on linear probing test to see which layer capture more linguistic structure 
information in their contextual representations. The output layers for classifying the POS tags are added on the different layers of BERT. We only train these layer's weights.

We treat BERT as a feature extractor to provide fixed pre-trained contextual embeddings. We set `requires_grad` false for BERT model. If you would like to fine-tune the whole model, just comment those lines.

In certain cases, rather than fine-tuning the entire pre-trained model end-to-end, it can be beneficial to obtain pre-trained contextual embeddings, which are fixed contextual representations of each input token generated from the hidden layers of the pre-trained model. This should also mitigate most of the out-of-memory issues.

We found that executing a  64 minibatch size trained with maximal sequence length is pretty slow. 
The maximal sequence length, in OntoNotes is 228, is usually an extreme case. We gain huge improvement on the runtime for a minibatch by using 63 to `max_seq_length` covering 99% of sequence length.

ALL the examples below use the dataset loading script for `OntoNotes`. If yo would like to run your dataset for sequence labeling, consult the official [tutorial](https://huggingface.co/docs/datasets/add_dataset.html).

### Train the official BERT model

The official Huggingface BERT for sequence labeling task using `BertForTokenClassification` class. 
The model consists of a pre-trained BERT with 12 layers, dropout and a classifier layer. The classifier heads on top of the last BERT's layer. 

The dimension of the hidden states  in BERT is 768. It uses 0.1 for dropout rate. Number of classses decides the output dimension. We calculate it from the dataset created by the loading script. We also freeze the BERT's weights.

To train the settings, you can run

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
 --do_predict \
 --learning_rate 1e-2 
```

### Train Linear-BERT  

Linear-BERT is the feature-based approach with BERT
and an architecture to extract the fixed contextual representations from the BERT. It aims to evaluate which layer captures linguistic structure information among the different layers. We freeze BERT’s weights and only train the classifier.

The custom model uses `bert-base-cased` as the base model follow by the dropout and linear classifier for labeling. You can specify the layer with the index from **0** to **12** to the argument `to_layer`. `0` indicates the embedding layer and the 12 BERT's layers are in the range of `1` to `12`. If you use a classifier on top of 12th BERT's layer, where you use `12` as the arugment. It is same as the standard BERT that `BertForTokenClassification` class creats for you. 

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

We support you to train custom model. The examples can be found in `custom-model-demo.py` and `bert-bilstm.py` under the model's folder. All you need to do is to modify the forward method with the addtional layers you would like try.

Run the custom model via the path `models/custom-model-demo.py`. Train your model by specifying `--model_name_or_path` to your script.


```python
 python run_pos.py \
 --model_name_or_path models/custom-model-demo.py \
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
 --logging_first_step \
 --logging_steps 5 \
 --learning_rate 1e-2 \
 --evaluation_strategy steps \
 --eval_steps 10 \
 --to_layer 2 
```

### Quicker training  

If you are developing the model or debugging it. 
The options `max_train_samples`, `max_vall_samples` and `max_test_samples` allow you to truncate the number of examples. It speeds up your experiements. They recieve digits digit format.

```python
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
 --logging_steps 20 \
 --learning_rate 1e-2
```

### Contact Information

For help or issues using the code, please submit a GitHub issue.
