echo off
set datasetname=%1

rmdir /s /q postprocess\result\%datasetname%
mkdir postprocess\result\%datasetname%
py -m postprocess.postprocess evaluate/result/%datasetname%/test_generations.txt postprocess/result/%datasetname%/test_generations.txt
py -m postprocess.postprocess evaluate/result/%datasetname%/test_label.txt postprocess/result/%datasetname%/test_label.txt

del evaluate\result\%datasetname%\smatch.csv
py -m evaluate.get_smatch postprocess/result/%datasetname%/test_generations.txt postprocess/result/%datasetname%/test_label.txt evaluate/result/%datasetname%/smatch.csv
