from functools import reduce

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def read_conll(p):
    with open(p, 'rt') as fd:
        sentence = []
        for line in fd:
            line = line.rstrip("\r\n")
            t = line.split("\t")
            assert len(t) in [0, 1, 3]
            if len(t) == 3:
                sentence.append(tuple(t))
            else:
                yield sentence.copy()
                sentence = []


class ConllDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = [tokenizer([x[1] for x in s], is_split_into_words=True, return_tensors='pt') for s in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataset = list(read_conll("sample.tsv"))
    batched_dataset = ConllDataset(dataset, tokenizer)
    print("Batch Size: %s" % len(batched_dataset))
