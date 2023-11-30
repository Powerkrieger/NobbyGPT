import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    # set directories
    working_dir = '../../'
    data_dir = f'{working_dir}data/'
    catalog_path = f'{data_dir}pg_catalog.csv'

    # read catalog into pandas
    data = pd.read_csv(catalog_path)
    deutsch = data.loc[data['Language'] == 'de']
    number_of_german_books = len(deutsch)
    print(f"Number of german books in gutenberg project: {number_of_german_books}")

    # Save the content to a .txt file

    missing_counter = 0
    downloaded_counter = 0
    # with open(save_file, 'a', encoding='utf-8') as file:
    for index, entry in deutsch.iterrows():
        value = entry['Text#']
        save_file = (f'{data_dir}'
                     f'{(entry["Authors"].strip()[:100] + "..") if len(entry["Authors"].strip()) > 100 else entry["Authors"].strip()}/'
                     f'{(entry["Title"][:100] + "..") if len(entry["Title"]) > 100 else entry["Title"]}.txt')
        print(f'book {index}, {entry["Title"]},by {entry["Authors"]}')
        if os.path.exists(save_file):
            downloaded_counter += 1
            continue
        plaintext_book_url = f'https://www.gutenberg.org/cache/epub/{value}/pg{value}.txt'
        # Send a GET request to the URL
        response = requests.get(plaintext_book_url)
        # if response.status_code != 200:
        #    missing_counter += 1
        #    continue
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        if soup.title is not None:
            missing_counter += 1
            continue

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        with open(save_file, 'w', encoding='utf-8') as file:
            try:
                # remove license and gutenberg info for use in model
                body_end = soup.get_text().split("*** START OF THE PROJECT GUTENBERG EBOOK ")[1]
                content = body_end.split("            *** END OF THE PROJECT GUTENBERG EBOOK")[0]
            except IndexError:
                file.close()
                print("well that was unfortunate")
                break
    print(f'Saved {number_of_german_books - missing_counter - downloaded_counter} of {number_of_german_books} german '
          f'books downloaded because {missing_counter} books were not available and {downloaded_counter} books were '
          f'already in possession.')
