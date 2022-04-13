import torch
import numpy as np
from tqdm.auto import tqdm
import pickle

from rouge import Rouge

from utils import preprocessing_tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args, s_model, c_model, train_dataloader, val_dataloader, s_optimizer, c_optimizer, loss_fn, loss_class, tokenizer, evaluation=False):
    print("Start training...\n")
    for epoch in range(args.epochs):
        # =======================================
        #               Training
        # =======================================
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9}")
        print("-"*70)

        total_loss, batch_loss, batch_counts = 0, 0, 0

        # summary text 저장
        summary_txt_list = []

        s_model.train()
        c_model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_summary, b_labels = tuple(t.to(device) for t in batch)

            s_model.zero_grad()
            c_model.zero_grad()

            outputs = s_model(b_input_ids, b_attn_mask, labels=b_summary)
            # logits = outputs['logits']
            loss_w = outputs['loss']

            summary_ids = s_model.generate(b_input_ids)
            summary_text = tokenizer.batch_decode(summary_ids.squeeze(), skip_special_tokens=True)
            summary_txt_list.append(summary_text)
            
            # summary_ids와 attention_mask 필요!
            train_input_ids, train_attention_mask = preprocessing_tokenize(summary_text, tokenizer)
            train_input_ids = train_input_ids.to(device)
            train_attention_mask = train_attention_mask.to(device)
            
            logits = c_model(train_input_ids, train_attention_mask)
            loss_c = loss_class(logits, b_labels) # classification loss
            preds = torch.argmax(logits, dim=1).flatten()
            loss_s = loss_fn(summary_text, step, preds, loss_w)

            batch_loss += loss_s.item()
            total_loss += loss_s.item()

            loss_c.backward()
            loss_s.backward()

            c_optimizer.step()
            s_optimizer.step()

            if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9}")
                batch_loss, batch_counts = 0, 0

        avg_train_loss = total_loss / len(train_dataloader)

        
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            val_loss, val_accuracy = evaluate(s_model, c_model, val_dataloader, loss_fn, loss_class, tokenizer)

            print(f"{epoch + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f}")
            print("-"*70)
        print("\n")
    print("Training complete!")


def evaluate(s_model, c_model, test_dataloader, loss_fn, loss_class, tokenizer):
    c_model.eval()
    s_model.eval()
    print(f"{'Val Loss':^10} | {'Val Acc':^9}")
    print("-"*70)
    
    val_accuracy = []
    val_loss = []
    summary_txt_list = []

    for step, batch in enumerate(test_dataloader):
        b_input_ids, b_attn_mask, b_summary, b_labels = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            outputs = s_model(b_input_ids, b_attn_mask, labels=b_summary)
            # logits = outputs['logits']
            loss_w = outputs['loss']

            summary_ids = s_model.generate(b_input_ids)
            
            summary_text = tokenizer.batch_decode(summary_ids.squeeze(), skip_special_tokens=True)
            summary_txt_list.append(summary_text)
            
            # summary_ids와 attention_mask 필요!
            train_input_ids, train_attention_mask = preprocessing_tokenize(summary_text, tokenizer)
            train_input_ids = train_input_ids.to(device)
            train_attention_mask = train_attention_mask.to(device)

            logits = c_model(train_input_ids, train_attention_mask)
            preds = torch.argmax(logits, dim=1).flatten()
        loss_s = loss_fn(summary_text, step, preds, loss_w)
        # rouge = Rouge()
        # rouge.get_scores(summary_text, b_summary, avg=True)

        filePath = './result/summary_result.txt'
        with open(filePath, 'wb', encoding='utf-8') as lf:
            pickle.dump(summary_txt_list, lf)
        

        val_loss.append(loss_s.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean()*100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    print(f"{val_loss:^10.6f} | {val_accuracy:^9.2f}")
    print("-"*70)
    print("\n")
    print("complete!")
    
    return val_loss, val_accuracy