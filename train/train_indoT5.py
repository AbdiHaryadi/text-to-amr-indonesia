import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from evaluate.evaluate_indoT5 import test_process
from utils.constants import AMR_TOKENS
from utils.data_utils import TextToAMRDataLoader, TextToAMRDataset
from utils.eval import generate
from utils.scoring import calc_corpus_bleu_score
from utils.utils_argparser import add_args
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from transformers.optimization import AdamW

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    ## init params
    set_seed(42)
    model_type = "indo-t5"
    batch_size = args.batch_size
    lr = args.lr
    eps = args.eps
    n_epochs = args.n_epochs
    num_beams = args.num_beams
    max_seq_len_amr = args.max_seq_len_amr
    max_seq_len_sent = args.max_seq_len_sent
    result_folder = args.result_folder
    DATA_FOLDER = args.data_folder
    if (args.resume_from_checkpoint):
        saved_model_folder_path = args.saved_model_folder_path


    cuda_used = False
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
        cuda_used = True
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    if (args.resume_from_checkpoint):
        print('resume from checkpoint')
        tokenizer = T5TokenizerFast.from_pretrained(os.path.join(saved_model_folder_path, 'tokenizer'))
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(saved_model_folder_path, 'model'))
    else:
        tokenizer = T5TokenizerFast.from_pretrained("Wikidepia/IndoT5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("Wikidepia/IndoT5-base", return_dict=True)

    #moving the model to device(GPU/CPU)
    model.to(device)

    # add new vocab (amr special tokens)
    new_tokens_vocab = {}
    new_tokens_vocab['additional_special_tokens'] = tokenizer.additional_special_tokens
    for idx, t in enumerate(AMR_TOKENS):
        new_tokens_vocab['additional_special_tokens'].append(t)

    num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
    print(f'added {num_added_toks} tokens')

    model.resize_token_embeddings(len(tokenizer))

    # load data

    train_amr_path = os.path.join(DATA_FOLDER, 'train.amr.txt')
    train_sent_path = os.path.join(DATA_FOLDER, 'train.sent.txt')

    dev_amr_path = os.path.join(DATA_FOLDER, 'dev.amr.txt')
    dev_sent_path = os.path.join(DATA_FOLDER, 'dev.sent.txt')

    # Change this, start here.
    train_dataset = TextToAMRDataset(train_sent_path, train_amr_path, tokenizer, 'train')
    dev_dataset = TextToAMRDataset(dev_sent_path, dev_amr_path, tokenizer, 'dev')

    train_loader = TextToAMRDataLoader(dataset=train_dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                                        batch_size=batch_size, shuffle=True)  
    dev_loader = TextToAMRDataLoader(dataset=dev_dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                                        batch_size=batch_size, shuffle=False)  

    print('len train dataset: ', str(len(train_dataset)))
    print('len dev dataset: ', str(len(dev_dataset)))

    print('len train dataloader: ', str(len(train_loader)))
    
    # define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        eps=eps
    )


    # train
    list_loss_train = []
    list_loss_dev = []
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)
    
        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        if not cuda_used:
            print("Warning: CUDA is not used during train")

        for i, batch_data in enumerate(train_pbar):
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

            outputs = model(input_ids=enc_batch, attention_mask=enc_mask_batch, decoder_input_ids=dec_batch, 
                        decoder_attention_mask=dec_mask_batch, labels=label_batch)
            loss, logits = outputs[:2]
            hyps = logits.topk(1, dim=-1)[1]
            
            loss.backward()
            
            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss
            
            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                    total_train_loss/(i+1), get_lr(optimizer)))
            
            optimizer.step()
            optimizer.zero_grad()

        list_loss_train.append(total_train_loss/len(train_loader))

        # eval per epoch
        model.eval()
        torch.set_grad_enabled(False)
        list_hyp, list_label = [], []
        
        total_dev_loss = 0

        pbar = tqdm(iter(dev_loader), leave=True, total=len(dev_loader))
        if not cuda_used:
            print("Warning: CUDA is not used during dev")
            
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

            outputs = model(input_ids=enc_batch, attention_mask=enc_mask_batch, decoder_input_ids=dec_batch, 
                        decoder_attention_mask=dec_mask_batch, labels=label_batch)
            loss, logits = outputs[:2]
            hyps = logits.topk(1, dim=-1)[1]
            
            batch_list_hyp = []
            batch_list_label = []
            for j in range(len(hyps)):
                hyp = hyps[j,:].squeeze()
                label = label_batch[j,:].squeeze()

                batch_list_hyp.append(tokenizer.decode(hyp[hyp != -100], skip_special_tokens=False))
                batch_list_label.append(tokenizer.decode(label[label != -100], skip_special_tokens=False))

            list_hyp += batch_list_hyp
            list_label += batch_list_label
            
            total_dev_loss += loss.item()
            pbar.set_description("(Epoch {}) DEV LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                    total_dev_loss/(i+1), get_lr(optimizer)))
            
        print("list_hyp:", *list_hyp, sep="\n")
        print("list_label:", *list_label, sep="\n")
        print()
            
        bleu = calc_corpus_bleu_score(list_hyp, list_label)
        print('bleu score on dev: ', str(bleu))

        list_loss_dev.append(total_dev_loss/len(dev_loader))

    ## TEST
    test_process(args, tokenizer, model)

    ## save loss data
    with open(os.path.join(result_folder, 'loss_data.tsv'), 'w') as f:
        f.write('train_loss\tval_loss\n')
        for i in range(n_epochs):
            f.write(f'{str(list_loss_train[i])}\t{str(list_loss_dev[i])}\n')

    ## save model
    # torch.save(model.state_dict(), os.path.join(result_folder, "indot5.th"))
    tokenizer.save_pretrained(os.path.join(result_folder, "tokenizer"))
    model.save_pretrained(os.path.join(result_folder, "model"))
    
    print(generate("saya mengetik makalah", model, tokenizer, num_beams, model_type, 'cpu'))    
