import urllib.request

if __name__ == "__main__":
    urllib.request.urlretrieve("https://raw.githubusercontent.com/elastic/hunspell/master/dicts/de_DE/de_DE.dic",
                               "de_DE.dic")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/elastic/hunspell/master/dicts/de_DE/de_DE.aff",
                               "de_DE.aff")
