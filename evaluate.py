import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

from model import Model
from utils.data_utils import ClassificationDataset, glue_processor, prepare_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, data_raw, mode='dev'):
    model.eval()
    test_data = ClassificationDataset(data_raw)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=test_data.collate_fn)
    preds = []
    labels = []
    epoch_pbar = tqdm(test_dataloader, desc="Evaluation", disable=False)
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, label_ids = batch
        with torch.no_grad():
            output = model(input_ids, segment_ids, input_mask)
        output = output.argmax(dim=1)
        output = output.tolist()
        label_ids = label_ids.tolist()
        preds = preds + output
        labels = labels + label_ids
        epoch_pbar.update(1)
    epoch_pbar.close()

    acc = cal_acc(preds, labels)
    print('Accuracy on ', mode, ' dataset: ', acc)
    return acc


def cal_acc(preds, labels):
    acc = sum([1 if p == l else 0 for p, l in zip(preds, labels)]) / len(labels)
    return acc


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)
    dev_data_raw = prepare_data(dev_examples,args.max_seq_len,tokenizer,labels)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.num_labels = len(labels)
    model = Model(model_config)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    evaluate(model,dev_data_raw,'dev')
    evaluate(model, test_data_raw, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='id', type=str)
    parser.add_argument("--data_dir", default='data/atis/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='outputs/model_best.bin', type=str)
    parser.add_argument("--max_seq_len", default=60, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
