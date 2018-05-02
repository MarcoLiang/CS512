import re
import codecs
from collections import Counter

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z -]', r'', text)
    tokens = text.split()
    return tokens


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

def remove_rare_words(text, counter):
    tokens = tokenize(text)
    return ' '.join([word for word in tokens if counter[word] >= 5])


print('loading training data and count tokens...')
file = 'training.txt'
with codecs.open(file, 'r', 'utf-8') as f:
    tokens_corpus = [tokenize(line.strip().split('\t')[1]) for line in f]
# counts, words_to_remove = count_tokens(tokens_corpus)
counter = toks_counter(tokens_corpus)
print('preprocess subset.txt...')
input_file = 'subset.txt'
output_file = 'cleaned_data.txt'
with open(output_file, 'w') as outf, codecs.open(input_file, 'r', 'utf-8') as inf:
    for line in inf:
        items = line.strip().split('\t')
        title = items[1]
        items[1] = remove_rare_words(title, counter)
        outf.write('\t'.join(items)+'\n')

# input_files = ['training.txt', 'validation.txt', 'test_set.txt']
# for input_file in input_files:
#     print('preprocess '+input_file+'...')
#     output_file = 'cleaned_'+input_file
#     with open(output_file, 'w') as outf, codecs.open(input_file, 'r', 'utf-8') as inf:
#         for line in inf:
#             items = line.strip().split('\t')
#             title = items[1]
#             items[1] = ' '.join(remove_rare_words(title, words_to_remove))
#             outf.write('\t'.join(items) + '\n')
