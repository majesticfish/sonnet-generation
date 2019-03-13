import re
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout

def get_token_dict():
    print("getting token")
    f = open("data/Syllable_dictionary.txt", 'r')
    lines = f.readlines()
    wordlist = []
    syllable_count = []
    wordlookup = {}
    for line in lines:
        l_split = line.split(" ")
        l = len(l_split)

        # create dictionary to map words to token number
        wordlookup[l_split[0].lower()] = len(wordlist)

        # map token number to word
        wordlist.append(l_split[0].lower())

        # grab all syllables
        syllable_count.append(l_split[1:])
    return (wordlist, syllable_count, wordlookup)

def process_shakespeare():
    wordlist, syllable_count, wordlookup = get_token_dict()
    print('processing')
    f = open("data/shakespeare.txt")
    lines = f.readlines();
    i = 0
    X = []
    while i < len(lines):
        line = lines[i]
        if line == '\n':
            i = i + 1
            continue
        try:
            int(line)
            i = i + 1
            continue
        except ValueError:
            pass
        x = []
        while i < len(lines) and lines[i] != '\n':
            line = lines[i]

            # remove extraenous puncutations
            line = line.replace('.','')
            line = line.replace(',','')
            line = line.replace('?','')
            line = line.replace('!','')
            line = line.replace(';','')
            line = line.replace(':','')
            line = line.replace('(','')
            line = line.replace(')','')
            line = line.strip('\n')

            words = line.split(' ')
            for word in words:
                if word == '':
                    continue
                if word.lower() in wordlookup:
                    x.append(wordlookup[word.lower()])
                else:
                    regex = re.compile('[^a-zA-Z]')
                    word = regex.sub('', word.lower())
                    if word in wordlookup:
                        x.append(wordlookup[word])
            i = i + 1
        X.append(x)
    return (X, wordlookup)

def get_data():
    f = open('data/shakespeare.txt')
    lines = f.readlines()
    text = ""
    for line in lines:
        try:
            int(line)
        except ValueError:
            text = text + line
    print("Text length %d" % len(text))
    v = sorted(set(text))
    print("Number of Unique Characters %d" % len(v))
    char2id = {u:i for i, u in enumerate(v)}
    id2char = np.array(v)
    text_as_int = np.array([char2id[c] for c in text])
    return text_as_int, text, v, char2id, id2char
