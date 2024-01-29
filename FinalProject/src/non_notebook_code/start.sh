#!/bin/bash
python ./finetune_llama.py;
#python ./inference_llama.py "denglish-weeve_lvl2" >../../Data/Generated_Samples/inference_lvl2/sample002.txt;
#python ./inference_llama.py "denglish-weeve_lvl3" >../../Data/Generated_Samples/inference_lvl3/sample002.txt;
#python ./inference_llama.py "denglish-weeve_lvl4" >../../Data/Generated_Samples/inference_lvl4/sample002.txt;
python ./inference_llama.py "textbook-german" >../../Data/Generated_Samples/textbook-german/sample002.txt;
