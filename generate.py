import argparse
import os
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from utils.eval import generate, set_seed
from utils.utils_argparser import add_args

if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    set_seed(42)

    saved_model_folder_path = args.saved_model_folder_path
    model_core_path = os.path.join(saved_model_folder_path, 'model')
    model_tokenizer_path = os.path.join(saved_model_folder_path, 'tokenizer')

    result = generate(
        text=input("Text: "),
        model=AutoModelForSeq2SeqLM.from_pretrained(model_core_path),
        tokenizer=T5TokenizerFast.from_pretrained(model_tokenizer_path),
        num_beams=args.num_beams,
        model_type="indo-t5"
    )
    print(result)
