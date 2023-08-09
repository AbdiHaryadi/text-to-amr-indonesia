import os
import torch
from utils.data_utils import TextToAMRDataset, TextToAMRDataLoader
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from tqdm import tqdm
import random
import numpy as np
import argparse

from utils.scoring import calc_corpus_bleu_score
from utils.utils_argparser import add_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def test_process(args, tokenizer, model):
    model_type = "indo-t5"
    batch_size = args.batch_size
    num_beams = args.num_beams
    max_seq_len_amr = args.max_seq_len_amr
    max_seq_len_sent = args.max_seq_len_sent
    result_folder = args.result_folder
    DATA_FOLDER = args.data_folder

    cuda_used = False
    if torch.cuda.is_available():
        cuda_used = True

    test_sent_path = os.path.join(DATA_FOLDER, 'test.sent.txt')
    test_amr_path = os.path.join(DATA_FOLDER, 'test.amr.txt')
    test_dataset = TextToAMRDataset(test_sent_path, test_amr_path, tokenizer, 'test')
    print('len test dataset:', str(len(test_dataset)))
    
    test_loader = TextToAMRDataLoader(dataset=test_dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                                        batch_size=batch_size, shuffle=False)
    
    model.eval()
    torch.set_grad_enabled(False)

    list_hyp, list_label = [], []

    pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
    if not cuda_used:
        print("Warning: CUDA is not used during test")

    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]

        enc_batch = torch.LongTensor(batch_data[0])
        dec_batch = torch.LongTensor(batch_data[1])
        enc_mask_batch = torch.FloatTensor(batch_data[2])
        dec_mask_batch = None
        label_batch = torch.LongTensor(batch_data[4])
        token_type_batch = None
        
        if cuda_used:
            enc_batch = enc_batch.cuda()
            dec_batch = dec_batch.cuda()
            enc_mask_batch = enc_mask_batch.cuda()
            label_batch = label_batch.cuda()

        # max_length is set to max_seq_len_amr, previously max_seq_len_sent
        hyps = model.generate(input_ids=enc_batch, attention_mask=enc_mask_batch, num_beams=num_beams, max_length=max_seq_len_amr, 
                            early_stopping=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)


        batch_list_hyp = []
        batch_list_label = []
        for j in range(len(hyps)):
            hyp = hyps[j]
            label = label_batch[j,:].squeeze()
        
            batch_list_hyp.append(tokenizer.decode(hyp[hyp != -100], skip_special_tokens=False))
            batch_list_label.append(tokenizer.decode(label[label != -100], skip_special_tokens=False)) # Filter padding, I guess.
        
        list_hyp += batch_list_hyp
        list_label += batch_list_label

    list_label = []
    for i in range(len(list_hyp)):
        if (i<5):
            print('sample: ', list_hyp[i], '----', test_dataset.data['amr'][i])
        list_label.append(test_dataset.data['amr'][i])
    
    ## BLEU SCORE
    bleu = calc_corpus_bleu_score(list_hyp, list_label)
    print('bleu score on test dataset: ', str(bleu))
    with open(os.path.join(result_folder, 'bleu_score_test.txt'), 'w') as f:
        f.write(str(bleu))

    ## save generated outputs
    with open(os.path.join(result_folder, 'test_generations.txt'), 'w') as f:
        for i in range(len(list_hyp)):
            e = list_hyp[i]
            f.write(e)
            if (i != len(list_hyp)-1):
                f.write('\n')
            
    ## save label 
    with open(os.path.join(result_folder, 'test_label.txt'), 'w') as f:
        for i in range(len(list_label)):
            e = list_label[i]
            f.write(e)
            if (i != len(list_label)-1):
                f.write('\n')

if __name__=='__main__':
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    set_seed(42)

    saved_model_folder_path = args.saved_model_folder_path

    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    tokenizer = T5TokenizerFast.from_pretrained(os.path.join(saved_model_folder_path, 'tokenizer'))
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(saved_model_folder_path, 'model'))
    print(tokenizer)
    print(model.config)

    #moving the model to device(GPU/CPU)
    model.to(device)

    test_process(args, tokenizer, model)