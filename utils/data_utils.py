import os

import torch
from torch.utils.data import Dataset
from transformers import DataProcessor, InputExample, logging

logger = logging.get_logger(__name__)


class IdProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train/intent_seq.in")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "train/intent_seq.out")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "dev/intent_seq.out")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "predict")

    def get_labels(self,data_dir):
        """See base class."""
        labels_list = self._read_tsv(os.path.join(data_dir, "vocab/intent_vocab"))
        labels = [label[0] for label in labels_list]
        return labels

    def _create_examples(self, lines_in,lines_out, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i,(line,out) in enumerate(zip(lines_in,lines_out)):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0][4:]
            text_b = ''
            label = None if set_type == "predict" else out[0].split(' ')[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PoemProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.csv")), "predict")

    def get_labels(self,data_dir):
        """See base class."""

        labels = ['1','2','3','4','5']
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i,line in enumerate(lines[1:]):
            line_list = line[0].split(',')
            guid = "%s-%s" % (set_type, i)
            text_a = line_list[1]
            text_b = ''
            label = None if set_type == "predict" else line_list[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class TrainingInstance:
    def __init__(self,example,max_seq_len):
        self.text_a = example.text_a
        self.text_b = example.text_b
        self.label = example.label
        self.max_seq_len = max_seq_len

    def make_instance(self,tokenizer,label_map):
        text_1 = tokenizer.tokenize(self.text_a)
        text_2 = tokenizer.tokenize(self.text_b)
        # TODO 判断长度越界

        text_1 = ["[CLS]"] + text_1 + ["[SEP]"]
        text_2 = text_2 + ["[SEP]"] if text_2 else text_2
        segment = [0] * len(text_1) + [1] * len(text_2)
        self.input_ = text_1 + text_2
        self.segment_id = segment
        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)

        if len(input_mask) < self.max_seq_len:
            self.input_id = self.input_id + [0] * (self.max_seq_len-len(input_mask))
            self.segment_id = self.segment_id + [0] * (self.max_seq_len-len(input_mask))
            input_mask = input_mask + [0] * (self.max_seq_len-len(input_mask))
        self.input_mask = input_mask
        self.label_id = label_map[self.label] if self.label else None


class ClassificationDataset(Dataset):
    def __init__(self,data,annotated=True):
        self.data = data
        self.len = len(data)
        self.annotated = annotated

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx]
    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in batch],dtype=torch.long) if self.annotated else None
        return input_ids,segment_ids,input_mask,label_ids

def prepare_data(examples,max_seq_len,tokenizer,labels):
    label_map = {label:idx for idx,label in enumerate(labels)}
    data = []
    for example in examples:
        instance = TrainingInstance(example,max_seq_len)
        instance.make_instance(tokenizer,label_map)
        data.append(instance)
    return data


glue_processor = {
    'id': IdProcessor(),
    'poem': PoemProcessor()
}