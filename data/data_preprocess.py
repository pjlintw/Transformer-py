"""Extract POS tags."""
import os
import argparse

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, default="sample.conll")
    parser.add_argument('--output_dir', type=str, default='./')

    return parser.parse_args()

def main():
    # Argument parser
    args = get_args()

    # Create output file
    write_file = os.path.join(args.output_dir, 'sample.tsv')
    wf = open(write_file, 'w')

    # Preprocess on `sample.conll`
    with open(args.dataset_name, 'r') as f:
        for line in f:
            # Ignore the document information
            if line[0] == '#':
                continue

            line_lst = line.split()
            if line == '\n':
                wf.write('*\n')
            elif len(line_lst) < 5:
                continue
            else:
                # Word position, word, pos tag from unpacked list 
                _,_,w_position, word, pos, *other = line_lst
                wf.write(f"{w_position}\t{word}\t{pos}\n")
            
    wf.close()

if __name__ == '__main__':
    main()
