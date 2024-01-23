import re
import matplotlib.pyplot as plt
import language_tool_python
from spellchecker import SpellChecker
import hunspell
import nltk


if __name__ == "__main__":
    filename = input("Enter the name of the file: ")
    # Configuration variables -> Change things here!
    tool = language_tool_python.LanguageTool('de-DE', config={ 'cacheSize': 1000, 'pipelineCaching': True, 'maxSpellingSuggestions': 1 }) # LanguageTool Setup
    spell = SpellChecker(language='de') # PySpellChecker Setup

    # For HunSpell you will need files from here: https://github.com/elastic/hunspell/tree/master/dicts/de_DE
    d = hunspell.HunSpell("de_DE.dic", "de_DE.aff") # Upload these two files from the provided GitHub URL into the instance!

    # Initialization of nltk
    nltk.download('words')
    eng_words = nltk.corpus.words.words()

    # Misc parameters
    text_preview_len = 256 # Length of the .txt preview
    vocab_hist_preview = 10 # Length of the vocabulary preview and german word preview
    # filename = 'Test.txt' # Name of the file to check

    print("Reading the textfile and previewing the first n characters :\n")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    words = text.split()
    words = [re.sub(r'[^a-zA-ZßäöüÄÖÜ]', '', word) for word in words]

    print("Lenth of file (words): ", len(words))
    print("Lenth of file (chars): ", len(text), "\n")
    print(text[:text_preview_len])

    print("Check with LanguageTool")
    matches = tool.check(text)
    print("LanguageTool analysis:")
    print("Number of spelling mistakes: ", len(matches))
    print("Error rate: ", (len(matches) / len(words)))

    print("Check with PySpellChecker")
    misspelled = spell.unknown(words)
    print("PySpellChecker analysis:")
    print("Misspelled Words: ", len(misspelled))
    print("Error Rate: ", (len(misspelled) / len(words)))

    print("Check with Hunspell")
    errors = []
    for word in words:
        if not d.spell(word):
            errors.append(word)

    print("HunSpell analysis:")
    print("Misspelled Words: ", len(errors))
    print("Error Rate: ", (len(errors) / len(words)))

    print("Build vocabulary and print size")
    words_v = [word.lower() for word in words]
    vocabulary_dict = dict.fromkeys(words_v)
    vocabulary = list(vocabulary_dict)
    print("Vocabulary size:", len(vocabulary))
    print("\nThe " + str(vocab_hist_preview) + " most used words:")

    print("Build histogram from vocabulary and preview the n most used words")
    vocab_hist = []
    for word in set(words_v):
        count = words_v.count(word)
        elem = (word, count)
        vocab_hist.append(elem)

    vocab_hist.sort(key=lambda x: x[1], reverse=True)

    for word, count in vocab_hist[:vocab_hist_preview]:
        print(f"{word}: {count}")

    print("German words and german vocabulary")
    ger_words = []  # for building the vocabulary later
    for word in words:
        if word not in eng_words:
            ger_words.append(word)

    ger_words_tr = []  # for the correct text output (inefficient, but whatever :)
    for voc in vocabulary:
        if voc not in eng_words:
            ger_words_tr.append(voc)

    print("Number of german words, according to nltk:", len(ger_words))
    print("German word rate:", (len(ger_words) / len(words)))
    print("\nThe " + str(vocab_hist_preview) + " first german words:")
    for entry in ger_words_tr[:vocab_hist_preview]:
        print(entry)

    print("\n")

    print("Build german vocabulary and print size")
    words_ger = [word.lower() for word in ger_words]
    vocabulary_dict_ger = dict.fromkeys(words_ger)
    vocabulary_ger = list(vocabulary_dict_ger)
    print("German vocabulary size:", len(vocabulary_ger))
    print("\nThe " + str(vocab_hist_preview) + " most used german words:")

    print("Build german histogram from vocabulary and preview the n most used words")
    vocab_hist_ger = []
    for word in set(words_ger):
        count = words_ger.count(word)
        elem = (word, count)
        vocab_hist_ger.append(elem)

    vocab_hist_ger.sort(key=lambda x: x[1], reverse=True)

    for word, count in vocab_hist_ger[:vocab_hist_preview]:
        print(f"{word}: {count}")
