"""Dataset loading script for OntoNotes v4.0 sparated by tab."""
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "OntoNotes 4.0"

class Ontonotes4Config(datasets.BuilderConfig):
    """BuilderConfig for OntoNotes 4.0"""

    def __init__(self, **kwargs):
        """BuilderConfig for OntoNotes 4.0.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Ontonotes4Config, self).__init__(**kwargs)


class Ontonotes4(datasets.GeneratorBasedBuilder):
    """OntoNotes 4.0."""
    
    BUILDER_CONFIGS = [
        Ontonotes4Config(name='ontonotes4', description="OntoNotes 4.0 dataset.")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "$",
                                "''",
                                ",",   
                                "-LRB-",
                                "-RRB-",
                                ".",
                                ":",
                                "ADD",
                                "AFX",
                                "CC",
                                "CD",
                                "DT",
                                "EX",
                                "FW",
                                "HYPH",
                                "IN",
                                "JJ",
                                "JJR",
                                "JJS",
                                "LS",
                                "MD",
                                "NFP",
                                "NN",
                                "NNP",
                                "NNPS",
                                "NNS",
                                "PDT",
                                "POS",
                                "PRP",
                                "PRP$",
                                "RB",
                                "RBR",
                                "RBS",
                                "RP",
                                "SYM",
                                "TO",
                                "UH",
                                "VB",
                                "VBD",
                                "VBG",
                                "VBN",
                                "VBP",
                                "VBZ",
                                "WDT",
                                "WP",
                                "WP$",
                                "WRB",
                                "XX",
                                "``"
                            ]
                        )
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        
        train_file = "./data/sample.train"
        dev_file = "./data/sample.dev"
        test_file ="./data/sample.test"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_file}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": train_file}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": train_file}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = list()
            pos_tags=list()
            for line in f:
                if line.startswith("*") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "pos_tags": pos_tags,
                        }
                        guid += 1
                        tokens=list()
                        pos_tags=list()
                else:
                    # line is tab separated
                    splits = line.split('\t')
                    tokens.append(splits[1])
                    pos_tags.append(splits[2].rstrip())
            # yield last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "pos_tags": pos_tags,
            }


if __name__ == "__main__":
    Ontonotes4(__name__)





























