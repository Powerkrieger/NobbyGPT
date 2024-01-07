# Englman

This project is part of a university course related to Large Language Models.
The preparation and Tutorial phase heavily relies on the nanoGPT project by Andrej Kaparthy.
The Final Project relies on llama-2 and tries to evaluate language learning in the context of LLMs.

# Soon to be Medium Blogpost:
## About the troubles of language learning: A large bilingual language model story

### General description of our project

Our project was born out of the idea of teaching a LLM different types of Language, so we toyed with the idea of having it learn "denglish" which is a mixture of german and english that the youth begins to use more heavily in recent days. From the topic came the idea to study a LLMs language learning aspect more closely, and that is what will be tried in this project.

In this project the baseline is a llama-2 fork from huggingface that is finetuned to be able to speak german.
We will try to take a llama-2 model ourselves and compare different types of language learning results with the baseline and themselves. The different approaches will include:
- wikipedia text that is diglot weaved using weeve.ie on 4 different intensity levels
- simmilar text but in a prompt form: try to have the model learn which words are actually different.
- use gutenberg literature and just translate words ourselves in different percentages (might be obsolete, weeve.ie does basically that)
- use german text

Evaluation metrics will be the spelling and grammar error rates as well as the output vocabulary size.

### Related work

The diglot weave method was coined by Robbins Burling in [1](1) accordind to [this](https://languagemixing.com/how-it-works) webpage. It describes switching out words from your main language into your target language in an attempt to just derive the meaning by context. If consistently done this should be able to teach a language and if the author is right also in a rather fast way. 

### Setup tutorial for our approach / evaluation

It is not all too easy to get into finetuning LLMs, even after they are pretty much made widely available. The problem in my opinion is still the relatively high prices for getting your hands on good hardware. If you are like me still afraid to use Cloud Services (or dont know how to) it is not too easy getting into. A good tutorial by Andrej Kaparthy teaches you the basics about LLMs but only in theory, not how to actually make use of them. 

For that you will have to set out and find a nice tutorial like we did: [finetune llama on your own data](https://github.com/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb)

Explains well how to get finetuning running if you have 24 Gb of Graphical RAM to spare. 


### Scientific evaluation
### Summary


[1]“Some Outlandish Proposals For Teaching Foreign Languages” by Robbins Burling (University of Michigan) <i>Language Learning 18</i> (June 1968)&nbsp;61-76.&nbsp;<a href="https://doi.org/10.1111/j.1467%E2%80%911770.1987.tb00390.x">doi:10.1111/j.1467‑1770.1987.tb00390.x</a>
