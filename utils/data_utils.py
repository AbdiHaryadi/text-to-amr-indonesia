import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

T5_PREFIX = "translate indonesia sentence to linearized AMR: "

class TextToAMRDataset(Dataset):
    """Class to load preprocessed text-AMR data.
    """
    def __init__(self, file_sent_path, file_amr_path, tokenizer, split,
                 file_level_path = None):
        temp_list_sent_input = []
        with open(file_sent_path, encoding='utf8') as f:
            temp_list_sent_input = f.readlines()
        list_sent_input = []
        for item in temp_list_sent_input:
            list_sent_input.append(item.strip().lower())  # lowercase for bart tokenizer
            
        temp_list_amr_output = []
        with open(file_amr_path, encoding='utf8') as f:
            temp_list_amr_output = f.readlines()
        list_amr_output = []
        for item in temp_list_amr_output:
            list_amr_output.append(item.strip().lower())
        
        if file_level_path is not None:
            raise NotImplementedError
        
        df = pd.DataFrame(list(zip(list_sent_input, list_amr_output)), columns = ['sent','amr'])
        self.with_tree_level = False
        self.data = df
        self.tokenizer = tokenizer
 
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sent, amr = data['sent'], data['amr']
       
        tokenize_sent = self.tokenizer.encode(sent, add_special_tokens=True)
        tokenize_amr = self.tokenizer.encode(amr, add_special_tokens=True)
        
        item = {'input':{}, 'output':{}}
        item['input']['encoded'] = tokenize_sent
        item['input']['raw'] = sent
        item['output']['encoded'] = tokenize_amr
        item['output']['raw'] = amr
        
        if (self.with_tree_level):
            raise NotImplementedError
            
        return item
    
    def _encode_tokens_with_tree_level(self, amr, level):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)

class TextToAMRDataLoader(DataLoader):
    """This dataloader class for T5, adapted from
    `https://github.com/indobenchmark/indonlg/blob/master/utils/data_utils.py`.
    """
    def __init__(self, max_seq_len_sent=384, max_seq_len_amr=512,
                 label_pad_token_id=-100, model_type='indo-t5', tokenizer=None,
                 with_tree_level=False, *args, **kwargs):
        if (with_tree_level):
            raise NotImplementedError
        
        super(TextToAMRDataLoader, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len_sent = max_seq_len_sent
        self.max_seq_len_amr = max_seq_len_amr
        
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        self.label_pad_token_id = label_pad_token_id
        self.with_tree_level = with_tree_level
        
        if model_type == 'indo-t5' or model_type == 'mT5':
            self.bos_token_id = tokenizer.pad_token_id
            self.t5_prefix =np.array(self.tokenizer.encode(T5_PREFIX, add_special_tokens=True))
            self.collate_fn = self._t5_collate_fn
        elif model_type == 'indo-bart':
            raise ValueError(f'Unknown model_type `{model_type}`')
            
    def _t5_collate_fn(self, batch):
        batch_size = len(batch)
        
        max_enc_len = min(self.max_seq_len_sent, max(map(lambda x: len(x['input']['encoded']), batch))  + len(self.t5_prefix))
        max_dec_len = min(self.max_seq_len_amr, max(map(lambda x: len(x['output']['encoded']), batch)) + 1)
        
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        level_batch = None
        
        for i, item in enumerate(batch):
            input_seq = item['input']['encoded']
            label_seq = item['output']['encoded']
            input_seq, label_seq = input_seq[:max_enc_len - len(self.t5_prefix)], label_seq[:max_dec_len - 1]
            
            # Assign content
            enc_batch[i,len(self.t5_prefix):len(self.t5_prefix) + len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + len(self.t5_prefix)] = 1
            dec_mask_batch[i,:len(label_seq) + 1] = 1
            
            # Assign special token to encoder input
            enc_batch[i,:len(self.t5_prefix)] = self.t5_prefix
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.bos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id            
        
        return enc_batch, dec_batch, enc_mask_batch, None, label_batch, level_batch
