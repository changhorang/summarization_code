import pandas as pd
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BertTokenizer

from model import Bart_summarization, BERTClassifier
from loss import loss_s
from utils import preprocessing_tokenize
from epoch import train, evaluate

def main(args):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # file load
    train_data = pd.read_csv('data/train_data.csv', sep='\t')
    train_input = train_data['article']
    train_summary = train_data['highlights']
    train_label = train_data['label']

    validation_data = pd.read_csv('data/valid_data.csv', sep='\t')
    validation_input = validation_data['article']
    validation_summary = validation_data['highlights']
    validation_label = validation_data['label']

    test_data = pd.read_csv('data/test_data.csv', sep='\t')
    test_input = test_data['article']
    test_summary = test_data['highlights']
    test_label = test_data['label']

    # tokenizing
    print('Tokenizing...')
    train_input_ids, train_attention_mask = preprocessing_tokenize(train_input, tokenizer)
    validation_input_ids, validation_attention_mask = preprocessing_tokenize(validation_input, tokenizer)
    test_input_ids, test_attention_mask = preprocessing_tokenize(test_input, tokenizer)

    train_summary_label, _ = preprocessing_tokenize(train_summary, tokenizer_bert)
    validation_summary_label, _ = preprocessing_tokenize(validation_summary, tokenizer_bert)
    test_summary_label, _ = preprocessing_tokenize(test_summary, tokenizer_bert)
    
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(train_label)
    validation_labels = torch.tensor(validation_label)
    test_labels = torch.tensor(test_label)

    # data_loader
    train_data = TensorDataset(train_input_ids, train_attention_mask, train_summary_label, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    val_data = TensorDataset(validation_input_ids, validation_attention_mask, validation_summary_label, validation_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

    test_data = TensorDataset(test_input_ids, test_attention_mask, test_summary_label, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    # train & evaluate
    s_model = Bart_summarization().to(device)
    c_model = BERTClassifier(freeze_bert=False).to(device)

    s_optimizer = torch.optim.Adam(s_model.parameters(), lr=args.lr, eps=1e-8)
    c_optimizer = torch.optim.Adam(c_model.parameters(), lr=args.lr, eps=1e-8)
    
    loss_fn = loss_s()
    loss_class = nn.CrossEntropyLoss()

    train(args, s_model, c_model, train_dataloader, val_dataloader, s_optimizer, c_optimizer, loss_fn, loss_class, tokenizer_bert, evaluation=True)
    evaluate(s_model, c_model, test_dataloader, loss_fn, loss_class, tokenizer_bert)

    # save model
    save_summary_model_name = f'{args.summary_model_name}.pt' # args
    save_classification_model_name = f'{args.classification_model_name}.pt' # args
    torch.save(s_model.state_dict(), os.path.join(args.save_summary_model_path, save_summary_model_name))
    torch.save(c_model.state_dict(), os.path.join(args.save_classification_model_path, save_classification_model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--summary_model_name', default='S_BART', type=str,
                        help='summary model save name')
    parser.add_argument('--classification_model_name', default='C_BERT', type=str,
                        help='classification model save name')
    parser.add_argument("--save_classification_model_path", type=str, default='./result/classfication/',
                        help="path to save classification model")
    parser.add_argument("--save_summary_model_path", type=str, default='./result/summary/',
                        help="path to save summary model")

    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='optimizer learning rate for train')
    parser.add_argument('--epochs', default=2, type=int, 
                        help='epochs for train')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='batch size for train')

    parser.add_argument('--seed', default=188, type=int,
                        help='Random seed for system')

    args = parser.parse_args()
    
    if not os.path.exists(args.save_summary_model_path):
        os.makedirs(args.save_summary_model_path)
    if not os.path.exists(args.save_classification_model_path):
        os.makedirs(args.save_classification_model_path)

    main(args)