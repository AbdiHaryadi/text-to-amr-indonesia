#!/bin/sh

SAVED_MODEL_FOLDER=$1
PREPROCESSING_METHOD=default

echo saved model folder ${SAVED_MODEL_FOLDER}
echo preprocessing method ${PREPROCESSING_METHOD}

####### predict all and evaluate #########
# cd evaluate
mkdir evaluate/result

# amr_simple_test
echo evaluate on data amr_simple_test
mkdir evaluate/result/amr_simple_test

python -m evaluate.evaluate_indoT5 --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder data/test/preprocessed_data/amr_simple_test --result_folder evaluate/result/amr_simple_test

# b-salah-darat
echo evaluate on data b-salah-darat
mkdir evaluate/result/b-salah-darat

python -m evaluate.evaluate_indoT5 --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder data/test/preprocessed_data/b-salah-darat --result_folder evaluate/result/b-salah-darat

# c-gedung-roboh
echo evaluate on data c-gedung-roboh
mkdir evaluate/result/c-gedung-roboh

python -m evaluate.evaluate_indoT5 --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder data/test/preprocessed_data/c-gedung-roboh --result_folder evaluate/result/c-gedung-roboh

# d-indo-fuji
echo evaluate on data d-indo-fuji
mkdir evaluate/result/d-indo-fuji

python -m evaluate.evaluate_indoT5 --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder data/test/preprocessed_data/d-indo-fuji --result_folder evaluate/result/d-indo-fuji

# f-bunuh-diri
echo evaluate on data f-bunuh-diri
mkdir evaluate/result/f-bunuh-diri

python -m evaluate.evaluate_indoT5 --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder data/test/preprocessed_data/f-bunuh-diri --result_folder evaluate/result/f-bunuh-diri

# g-gempa-dieng
echo evaluate on data g-gempa-dieng
mkdir evaluate/result/g-gempa-dieng

python -m evaluate.evaluate_indoT5 --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder data/test/preprocessed_data/g-gempa-dieng --result_folder evaluate/result/g-gempa-dieng
