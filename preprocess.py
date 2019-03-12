import re

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
        wordlist.append(l_split[0].lower())
        wordlookup[l_split[0].lower()] = len(wordlist) - 1
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


wordlist, syllable_count, wordlookup = get_token_dict()
print("WORDLIST")
print(wordlist)
print("SYLLABLE_COUNT")
print(syllable_count)
print("WORDLOOKUP")
print(wordlookup)
print(process_shakespeare())