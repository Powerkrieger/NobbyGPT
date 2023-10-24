import requests
from bs4 import BeautifulSoup

# URL of Goethe's Author page on Project Gutenberg
url = 'https://www.gutenberg.org/ebooks/author/586'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, 'html.parser')

# Find all links on the page
links = soup.find_all('a')

# List to store the URLs of Goethe's works
goethe_works_urls = []

# Extract the URLs of Goethe's works from the links
for link in links:
    href = link.get('href')
    if href and href.startswith('/ebooks/') and href[8:].isdigit():
        goethe_works_urls.append('https://www.gutenberg.org' + href)

# Function to download and save a book
def download_book(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.text.strip()
    content = soup.find('body').text.strip()

    # Save the content to a .txt file
    with open(f'{title}.txt', 'w', encoding='utf-8') as file:
        file.write(content)
    print(f'Saved: {title}.txt')

# Download and save each of Goethe's works
for work_url in goethe_works_urls:
    download_book(work_url)
