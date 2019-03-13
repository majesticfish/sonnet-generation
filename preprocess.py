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

def syllable_lookup():
    '''Helper to create a dictionary with key: word, value: syllables (int)'''
    wordlist, syllable_count, wordlookup = get_token_dict()
    syllable_dict = {}
    
    # syllable_count is a list of lists
    for i in range(len(wordlist)):
        if syllable_count[i][0][0] == 'E':
            syllable_dict[wordlist[i]] = int(syllable_count[i][1][0])
        else:
            syllable_dict[wordlist[i]] = int(syllable_count[i][0][0])
            
    return syllable_dict
    
    

wordlist, syllable_count, wordlookup = get_token_dict()
'''
print("WORDLIST")
print(wordlist)
print("SYLLABLE_COUNT")
print(syllable_count)
print("WORDLOOKUP")
print(wordlookup)
print(process_shakespeare())'''
print(syllable_lookup())

def get_data():
    X, word_to_id = process_shakespeare()
    X_train = []
    X_test = []
    for i in range(len(X)):
        x = X[i]
        # arbitrarily split data set into training and set
        if i < 130:
            for word in x:
                X_train.append(word)
        if i >= 130:
            for word in x:
                X_test.append(word)
    vocab = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    return X_train, X_test, vocab, reversed_dictionary
