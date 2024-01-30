#!/bin/bash
# python ./finetune_llama.py;
python ./inference_llama.py "denglish-weeve_lvl2" >../../Data/Generated_Samples/inference_lvl2/sample003.txt;
python ./inference_llama.py "denglish-weeve_lvl3" >../../Data/Generated_Samples/inference_lvl3/sample003.txt;
python ./inference_llama.py "denglish-weeve_lvl4" >../../Data/Generated_Samples/inference_lvl4/sample003.txt;
python ./inference_llama.py "textbook-german_with-loss" >../../Data/Generated_Samples/textbook-german/sample003.txt;
python ./spellchecking_llama.py;