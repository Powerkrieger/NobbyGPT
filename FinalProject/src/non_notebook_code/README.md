### 1.
```
pip install --upgrade virtualenv
```
### 2.
```
python -m venv <newenv>
```
### 3.
```
source <newenv>/bin/activate
```
### 4.
```
pip install -r requirements.txt
```

#### Requirements file
A requirements.txt file can be created using 
```
pip freeze > requirements.txt
```


#### steps to setup spellchecking
Because hunspell and languagetool are being difficult, run these in a terminal, and you should be good to go. 
This is a good example of read the readme, because you won't get the right debug messages
```
sudo apt install build-essential python3-dev libhunspell-dev
```

```
pip install hunspell
```

```
sudo apt install openjdk-17-jre-headless
```
