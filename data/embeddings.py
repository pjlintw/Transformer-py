import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from data.dataset import ConllDataset, read_conll


class BertEmbeddings(Dataset):
    def __init__(self, batches, model):
        self.data = []
        for sentence in tqdm(batches):
            self.data.append(model(**sentence)[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    tokens = ConllDataset(read_conll("sample.tsv"), tokenizer)
    embeddings = BertEmbeddings(tokens, model)
    torch.save(embeddings, "embeddings.torch.zip")
