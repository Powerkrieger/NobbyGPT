import random
import json

with open("German_Learning_Textbook/ger_book.txt", "r", encoding='utf-8') as f:
    text = f.read()

length = 800

with open("German_Learning_Textbook/ger_book.json", "w", encoding='utf-8') as out:
    encoder = json.JSONEncoder()

    for i in range(0, len(text), length):
        start = random.randrange(len(text))
        input = text[start: start + length]

        out.write(encoder.encode({"input": input}))
        out.write("\n")
