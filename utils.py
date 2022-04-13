import numpy as np
import torch

def preprocessing_tokenize(data, tokenizer):
    input_ids = []
    attention_masks = []
    MAX_LEN = 64

    for sent in data:
        encoded_sent = tokenizer.encode_plus(text=sent,
                                            add_special_tokens=True, 
                                            max_length=MAX_LEN, 
                                            pad_to_max_length=True,
                                            #return_tensors='pt',
                                            return_attention_mask=True)

        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks