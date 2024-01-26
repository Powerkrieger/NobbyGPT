#!/bin/bash
python ./finetune_llama.py;
python ./inference_llama.py "denglish-weeve_lvl2" >inference_lvl2/sample002.txt;
python ./inference_llama.py "denglish-weeve_lvl3" >inference_lvl3/sample002.txt;
