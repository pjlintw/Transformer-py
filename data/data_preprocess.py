"""Extract POS tags and split dataset."""
import os
import argparse
import random

from pathlib import Path

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, default="sample.conll")
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--eval_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=1000)

    return parser.parse_args()


def build_examples(data_file):
    """Load examples from `sample.conll`

    Returns:
      examples: List of tuple contains (1) position sequence,
                (2) word sequence, (3) pos taggs.
    """
    # List of tuple contain (w_pos, word, pos)
    examples = list()
    with open(data_file, 'r') as f:
        w_position_list = list()
        word_list = list()
        pos_list = list()

        for line in f:
            # Ignore the document information
            if line[0] == '#':
                continue

            line_lst = line.split()
            if line == '\n':
                assert len(w_position_list) == len(word_list)
                # Append example
                example = (w_position_list, 
                           word_list,
                           pos_list)
                examples.append(example)

                w_position_list = list()
                word_list = list()
                pos_list = list()

            elif len(line_lst) < 5:
                continue
            else:
                # Word position, word, pos tag from unpacked list 
                _,_,w_position, word, pos, *other = line_lst
                w_position_list.append(w_position)
                word_list.append(word)
                pos_list.append(pos)
    return examples


def save_example_to_file(examples, output_file):
    """Save examples.

    Args:
      examples: List of examples
      output_file: wirte file.
    """
    with open(output_file, 'w') as wf:
        for (position_list, word_list, tag_list) in examples:
            # ["0","1"], ["Hello", "World"], ["O","O"]
            for w_position, w, tag in zip(position_list, word_list, tag_list):
                # Separate by tab
                wf.write(f"{w_position}\t{w}\t{tag}\n")
            wf.write('*\n')
        


def main():
    # Argument parser
    args = get_args()
    SEED = 49

    
    # Collect examples from `sample.conll`
    examples = build_examples(data_file=args.dataset_name)
    num_examples = len(examples)
    print("Loading {} examples".format(num_examples))

    # Shuffle 
    random.Random(SEED).shuffle(examples)
    print("Seed {} is used to shuffle examples".format(SEED))
    
    # Write `sample.tsv`
    write_file = Path(args.output_dir, "sample.tsv")
    save_example_to_file(examples=examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(num_examples, write_file))
    

    ### Train, validation, test splits. ###
    n_eval = int(args.eval_samples)
    n_test = int(args.test_samples)

    # Spliting datasets
    train_examples = examples[:-n_eval-n_test]
    eval_start = -(n_eval+n_test)
    eval_examples = examples[eval_start:-n_test]
    test_examples = examples[-n_test: ]
    
    # Write `sample.train`
    write_file = Path(args.output_dir, "sample.train")
    save_example_to_file(examples=train_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(train_examples), write_file))


    # Write `sample.dev`
    write_file = Path(args.output_dir, "sample.dev")
    save_example_to_file(examples=eval_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(eval_examples), write_file))
    

    # Write `sample.test`
    write_file = Path(args.output_dir, "sample.test")
    save_example_to_file(examples=test_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(test_examples), write_file))





if __name__ == '__main__':
    main()
