"""Run the pretrained BERT or custom model on POS taggins task."""
import logging
import sys
import inspect
import importlib.util

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    BertModel,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    BertTokenizer, 
    AutoTokenizer, 
    BertTokenizerFast, 
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

@dataclass
class ModelArguments:
    """Arguments to the pretrained model."""
    model_name_or_path: str = field(
        default="bert-base-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models or custom model"}
    )

    config_name: Optional[str] = field(
        default="bert-base-cased", metadata={"help": "Pretrained config name or path."}
    )

    tokenizer_name: Optional[str] = field(
        default="bert-base-cased",
        metadata={"help": "Pretrained tokenizer name or path."}
    )

@dataclass
class DataTrainingArguments:
    """Arguments to data involved."""
    task_name: Optional[str] = field(
        default="pos", 
        metadata={"help": "The name of the task (ner, pos...)."}
        )

    dataset_script: Optional[str] = field(
        default="ontonotes_v4.py", 
        metadata={"help": "Dataset loading script."}
        )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "maximal length to be padded."}
        )
    
    return_tag_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return the tag levels during evaluation."}
        )


def get_label_list(labels):
    """"Get label list from the `pos_tags`.

    Args:
      labels: <class 'datasets.features.Sequence'> for tags.

    Return:
      label_lst: List of labels.
    """
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def compute_metrics(p):
    """Compute the evaluation metric for POS tagging."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if data_args.return_tag_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def main():
    # Parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    ########## Load dataset from script. ##########
    # 'ontonotes_v4.py'
    datasets = load_dataset(data_args.dataset_script)
    
    ### Access column names and features ###
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features

    # In the event the labels are not a `Sequence[ClassLabel]`,
    # we will need to go through the dataset to get the unique labels.
    if isinstance(features["pos_tags"].feature, ClassLabel):
        label_list = features["pos_tags"].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"]["pos_tags"])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    

    ########## Load pre-trained or custom model, tokenizer and config ##########
    # BertConfig
    config = AutoConfig.from_pretrained(
        model_args.config_name,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=None,
        revision="main",
        use_auth_token=True if False else None,
    )

    # BERTTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=None,
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )

    # Create custom model or pretained BERT
    if ".py" == model_args.model_name_or_path[-3:]:
        spec = importlib.util.spec_from_file_location("module.name",  model_args.model_name_or_path)
        module_name = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_name)
        # Creating custom model
        model = module_name.model()(config)
        print(f"Creating custom model in {model_args.model_name_or_path}")
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in "bert-base-cased"),
            config=config,
            cache_dir=None,
            revision="main",
            use_auth_token=True if False else None,
        )
        # Freeze BERT
        for param in model.base_model.parameters():
            param.requires_grad = False


    # Metrics
    metric = load_metric("seqeval")

    def tokenize_fn(examples):
        """Tokenize the input sequence and align the label.

        `input_ids` and `label_ids` will be added in the feature example (dict).
        They are required for the forward and loss computation.

        Addtionally. `-100` in `label_ids` is assigned to segmented tokens
        and to speical tokens in BERT. Loss function will ignore them.

        Args:
          Examples: dict of features:
                    {"tokens": [AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06'],
                     "pos_tags": [22, 6, 22, 22, 23, 11]}

        Return:
          tokenized_inputs: dict of futures including two 
                            addtional feature: `input_ids` and `label_ids`.
        
        Usages:
        >>> tokenized_dataset = datasets.map(tokenize_fn,
        >>>                                  batched=True)
            # Check whether aligned.
        >>> for example in tokenized_dataset:
                tokens = example['tokens']
                input_ids = example['input_ids']
                tokenized_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                label_ids = example['label_ids'] # aligned to max length 
                print(tokens)
                print(tokenized_tokens)
                print(input_ids)

        [ 'SOCCER' ] # token
        [ [CLS], 'S', '##OC, '##CE', '##R', [SEP] ] #converted_tokens 
        [ -100 ,  4 , -100,  -100, -11] # label_ids
        """
        token_col_name = 'tokens'
        label_col_name = 'pos_tags'

        # will be added in the dict
        tokenized_inputs = tokenizer(examples[token_col_name],  
                                     padding="max_length",
                                     truncation=True,
                                     is_split_into_words=True,
                                     max_length=data_args.max_seq_length)
        
        # Create label sequence
        # The `word_idx` in is used to map tokens to actual word .
        # tokenied_token: [CLS, Ha, ##LLO,  PAD,  PAD]
        # `word_ids`:     [None, 1,     1, None, None]
        # `label_ids`:    [-100, 4,     4, -100, -100]
        labels = list()
        for i, label in enumerate(examples["pos_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # None are for [CLS], [SEP] and [PAD] tokens.
                if word_idx is None:
                    label_ids.append(-100)
                # Set label for the word once
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    #label_ids.append(label)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        # Add `labels` sequence for loss computation
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = datasets.map(
            tokenize_fn,
            batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)


    ########## Train, evaluate and test with Trainer ##########
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["test"] if True else None,
        eval_dataset=tokenized_dataset["validation"] if True else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    last_checkpoint=None
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

        trainer.log_metrics("eval", results)
        trainer.save_metrics("eval", results)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_dataset = tokenized_datasets["test"]
        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    return results


if __name__ == '__main__':
    main()











