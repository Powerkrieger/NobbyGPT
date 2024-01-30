import json
import re

import chardet
import hunspell
import language_tool_python
import nltk
import pandas as pd
from spellchecker import SpellChecker


def read_sorted_dictionary(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as file:
        dictionary_set = {line.strip() for line in file}
    return dictionary_set, encoding


# Function to check if a word is in the dictionary
def is_word_in_dictionary(word, dictionary):
    return word in dictionary


def check_text_file(file_path, dictionary, encoding):
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
        words = content.split()
        histogram = {}
        for word in words:
            # Remove punctuation if needed
            word = word.strip('.,!?()[]{}"\'')
            if is_word_in_dictionary(word, dictionary):
                histogram[word] = histogram.get(word, 0) + 1
    return histogram


def german_dictionary():
    dict_file_path = '../../Data/dict/ngerman'
    with open(dict_file_path, 'rb') as raw_file:
        result = chardet.detect(raw_file.read())
    encoding = result['encoding']
    # Replace 'your_sorted_dictionary.txt' with the actual path to your sorted dictionary file
    dictionary, encoding = read_sorted_dictionary(dict_file_path, encoding)
    return dictionary, encoding


def clean_invalid_characters(text):
    # Replace invalid control characters
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return cleaned_text

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.txt'):
            # Read the whole text for .txt files
            text = file.read()
        else:
            # Initialize an empty string to concatenate text from each JSON object
            text = ''
            for line in file:
                try:
                    json_data = json.loads(line)
                    # Concatenate values from each JSON object
                    text += ' '.join(str(value) for value in json_data.values()) + ' '
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

    return text.strip()  # Remove trailing whitespace


def prepare_text(filename, text_preview_len):
    text = read_file(filename)
    words = text.split()
    # remove everything that is not a letter
    words = [re.sub(r'[^a-zA-ZßäöüÄÖÜ]', '', word) for word in words]
    print("Length of file (words): ", len(words))
    print("Length of file (chars): ", len(text), "\n")
    print(f"{text_preview_len} characters: {text[:text_preview_len]}")
    print("\n---------------------------------------------------------\n")

    return text, words


def language_tool_check(text, words):
    # Languagetool setup
    tool = language_tool_python.LanguageTool('de-DE',
                                             config={'cacheSize': 1000,
                                                     'pipelineCaching': True,
                                                     'maxSpellingSuggestions': 1})

    matches = tool.check(text)
    print("LanguageTool analysis:")
    print("Number of spelling mistakes: ", len(matches))
    print("Error rate: ", (len(matches) / len(words)))
    print("\n---------------------------------------------------------\n")
    return len(matches), (len(matches) / len(words))


def pyspellchecker_check(words):
    # PySpellChecker Setup
    pyspell = SpellChecker(language='de')

    misspelled = pyspell.unknown(words)
    print("PySpellChecker analysis:")
    print("Misspelled Words: ", len(misspelled))
    print("Error Rate: ", (len(misspelled) / len(words)))
    print("\n---------------------------------------------------------\n")
    return len(misspelled), (len(misspelled) / len(words))


def hunspell_check(words):
    # For HunSpell you will need files from here: https://github.com/elastic/hunspell/tree/master/dicts/de_DE
    d = hunspell.HunSpell("../../Data/dict/de_DE.dic",
                          "../../Data/dict/de_DE.aff")

    errors = []
    for word in words:
        if not d.spell(word):
            errors.append(word)

    print("HunSpell analysis:")
    print("Misspelled Words: ", len(errors))
    print("Error Rate: ", (len(errors) / len(words)))
    print("\n---------------------------------------------------------\n")
    return len(errors), (len(errors) / len(words))


def print_histogram(vocab_hist, vocab_hist_preview):
    for word, count in vocab_hist[:vocab_hist_preview]:
        print(f"{word}: {count}")


def build_vocab(words, vocab_hist_preview):
    vocab_hist = {}
    for word in words:
        vocab_hist[word.lower()] = vocab_hist.get(word.lower(), 0) + 1
    vocab_hist = dict(sorted(vocab_hist.items(), key=lambda item: item[1], reverse=True))
    print("Vocabulary size of sample (all words used, lowercase):", len(vocab_hist))
    print(f"The {vocab_hist_preview} most used words:")
    for entry in list(vocab_hist.keys())[:vocab_hist_preview]:
        print(entry)
    print("\n---------------------------------------------------------\n")
    return vocab_hist


def check_german_vocab(words, eng_dict, ger_dict, vocab_hist_preview):
    ger_hist = {}
    for word in words:
        if word in ger_dict:
            ger_hist[word] = (ger_hist.get(word, 0) + 1)
    print(f"{len(ger_hist)} different words that match the german dictionary")

    n_ger_words = sum(ger_hist.values())

    print("Number of german words, according to nltk:", n_ger_words)
    print("German word rate:", (n_ger_words / len(words)))
    print("\nThe " + str(vocab_hist_preview) + " first german words:")
    ger_hist = dict(sorted(ger_hist.items(), key=lambda item: item[1], reverse=True))
    for entry in list(ger_hist.keys())[:vocab_hist_preview]:
        print(entry)
    print("\n---------------------------------------------------------\n")

    ger_not_engl = {}
    for key, value in ger_hist.items():
        if key not in eng_dict:
            ger_not_engl[key] = value
    n_only_ger_words = sum(ger_not_engl.values())

    print(f"{len(ger_not_engl)} different words that are only german (not english)")
    print("Number of german words, according to nltk:", n_only_ger_words)
    print("German word rate:", (n_only_ger_words / len(words)))
    print("\nThe " + str(vocab_hist_preview) + " first german words:")
    ger_not_engl = dict(sorted(ger_not_engl.items(), key=lambda item: item[1], reverse=True))
    for entry in list(ger_not_engl.keys())[:vocab_hist_preview]:
        print(entry)
    print("\n---------------------------------------------------------\n")
    return ger_hist, n_ger_words, (n_ger_words / len(words)), ger_not_engl, n_only_ger_words, (n_only_ger_words / len(words))


if __name__ == "__main__":
    # filename = input("Enter the name of the file: ")
    filenames = []
    filenames.append("../../Data/Generated_Samples/german_baseline/sample001.txt")  # TODO sample002.txt
    filenames.append("../../Data/Generated_Samples/inference_lvl2/sample003.txt")
    filenames.append("../../Data/Generated_Samples/inference_lvl3/sample003.txt")
    filenames.append("../../Data/Generated_Samples/inference_lvl4/sample003.txt")
    filenames.append("../../Data/Generated_Samples/textbook-german/sample003.txt")
    filenames.append("../../Data/Training_Data/WeeveIE_Wikipedia/WeeveLVL2_J.json")
    filenames.append("../../Data/Training_Data/WeeveIE_Wikipedia/WeeveLVL3_J.json")
    filenames.append("../../Data/Training_Data/WeeveIE_Wikipedia/WeeveLVL4_J.json")
    filenames.append("../../Data/Training_Data/German_Learning_Textbook/ger_book.json")



    # Initialization of nltk
    # nltk.download('words')
    eng_words = nltk.corpus.words.words()
    ger_dict, encoding = german_dictionary()


    evaluation_data = []
    for filename in filenames:
        eval_entry = {}
        eval_entry["filename"] = filename

        text, words = prepare_text(filename, text_preview_len=256)
        eval_entry["n_words(orig)"] = len(words)
        eval_entry["n_chars(orig)"] = len(text)

        vocab_hist = build_vocab(words, vocab_hist_preview=10)
        eval_entry["vocab(orig)"] = len(vocab_hist)

        n_matches, match_rate = language_tool_check(text, words)
        eval_entry["spelling_errors(lantool)"] = n_matches
        eval_entry["spelling_error_rate(lantool)"] = match_rate

        n_misspelled, error_rate = pyspellchecker_check(words)
        eval_entry["spelling_errors(pyspell)"] = n_misspelled
        eval_entry["spelling_error_rate(pyspell)"] = error_rate

        n_hunspell_errors, hunspell_error_rate = hunspell_check(words)
        eval_entry["spelling_errors(hunspell)"] = n_hunspell_errors
        eval_entry["spelling_error_rate(hunspell)"] = hunspell_error_rate

        ger_hist, ger_words, ger_word_rate, ger_not_engl, minus_engl_ger, minus_engl_ger_rate = check_german_vocab(words, eng_words, ger_dict, vocab_hist_preview=10)
        eval_entry["ger_words"] = ger_words
        eval_entry["ger_word_examples"] = list(ger_hist.keys())[:10]
        eval_entry["ger_word_rate"] = ger_word_rate
        eval_entry["minus_engl_ger"] = minus_engl_ger
        eval_entry["only_ger_word_examples"] = list(ger_not_engl.keys())[:10]
        eval_entry["minus_engl_ger_rate"] = minus_engl_ger_rate

        evaluation_data.append(eval_entry)

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(evaluation_data)

    # Specify the output Excel file path
    excel_file_path = 'evaluation_results.xlsx'

    # Write the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)

    print(f"Excel file '{excel_file_path}' created successfully.")
