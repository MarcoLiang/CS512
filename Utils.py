import re
from collections import Counter


def tokenize(title):
    title = title.lower()
    title = re.sub(r'[^a-z -]', r'', title)
    toks = title.split()
    return toks

def toks_counter(corpus):
    '''
    :param corpus: a list of list
    :param threshold: words with frequent < threshold should be removed
    :return: A counter
    '''
    counter = Counter()
    for text in corpus:
        counter.update(text)
    return counter

def remove_low_freq_words(text, counter, threshold=5):
    tokens = tokenize(text)
    return ' '.join([word for word in tokens if counter[word] >= 5])





