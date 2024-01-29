import chardet


def read_sorted_dictionary(file_path):
    with open(file_path, 'r', encoding=encoding) as file:
        dictionary_set = {line.strip() for line in file}
    return dictionary_set, encoding

# Function to check if a word is in the dictionary
def is_word_in_dictionary(word):
    return word in dictionary


# Function to check a text file against the dictionary
def check_text_file(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
        words = content.split()
        histogram = {}
        for word in words:
            # Remove punctuation if needed
            word = word.strip('.,!?()[]{}"\'')
            if is_word_in_dictionary(word):
                histogram[word] = histogram.get(word, 0) + 1
    return histogram


dict_file_path = '../../Data/dict/ngerman'
sample_file_path = '../../Data/Generated_Samples/inference_lvl2/sample002.txt'
with open(dict_file_path, 'rb') as raw_file:
    result = chardet.detect(raw_file.read())

encoding = result['encoding']

# Replace 'your_sorted_dictionary.txt' with the actual path to your sorted dictionary file
dictionary, encoding = read_sorted_dictionary(dict_file_path)

# Replace 'your_text_file.txt' with the actual path to your text file
histogram = check_text_file(sample_file_path, encoding)

sorted_dict = {key: histogram[key] for key in sorted(histogram.keys())}

for key, value in histogram.items():
    print(f"{key:15}: {value}")