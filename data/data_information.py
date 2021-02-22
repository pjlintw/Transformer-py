"""Get information about the POS tags."""
import re
import os
import argparse
from collections import defaultdict
import math

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, default="sample.tsv")
    parser.add_argument('--output_dir', type=str, default='./')

    return parser.parse_args()

def main():
    # Argument parser
    args = get_args()

    # Init variables for counting
    num_sentences = 0
    num_tags = 0
    cur_sentence_length = 0
    sent_length_lst = list()
    tag2freq = defaultdict(int)

    # Preprocess tags in `sample.tsv`
    with open(args.dataset_name, 'r') as f:    
        for idx, line in enumerate(f):
            line_lst = line.strip().split()
            # Get sentence length when encounting `*`
            if line[0] == '*':
                num_sentences += 1
                sent_length_lst.append(cur_sentence_length+1)
                continue 

            # Counting
            assert len(line_lst) == 3 
            w_position, word, tag = line_lst
            tag2freq[tag] += 1
            num_tags += 1
            cur_sentence_length = int(w_position)

    # Maximum, min and mean sequence length
    max_len = max(sent_length_lst)
    min_len = min(sent_length_lst)
    mean_len = sum(sent_length_lst) / len(sent_length_lst)

    # Create output file
    write_file = os.path.join(args.output_dir, 'sample.info')

    with open(write_file, 'w') as wf:
        # Write max, min, mean sentence length and number of sent
        wf.write(f'Max sequence length: {max_len}\n')
        wf.write(f'Min sequence length: {min_len}\n')
        wf.write(f'Mean sequence length: {mean_len}\n')
        wf.write(f'Number of sequence: {num_sentences}\n\n')
        wf.write('Tags:\n')

        # Write POS tag and its percentage of the words
        tag_percentage = [ (k, '%.2f'%(v/num_tags*100)) for k,v in tag2freq.items() ]
        tag_percentage = sorted(tag_percentage, key=lambda x: x[0])
        # Sort by tag name
        for (tag, perc)  in sorted(tag_percentage, key=lambda x: x[0]):
            wf.write(f'{tag : <5}\t{perc}%\n')    

if __name__ == '__main__':
    main()
