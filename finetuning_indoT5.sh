mkdir -p train/result
python -m train.train_indoT5 --model_type indo-t5 --n_epochs 3 --data_folder data/preprocessed_data/linearized_penman
