import sklearn
import re
import codecs
from Utils import *
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
regex = re.compile('[^a-z\ -]')


class PublicationVenueSuggestion:

    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()

    def data_cleaning(self):
        print('Loading training.txt...')
        with codecs.open('data/training.txt', 'r', 'utf-8') as f:
            corpus_toks = [tokenize(line.strip().split('\t')[1]) for line in f]
        counter = toks_counter(corpus_toks)

        input_files = ['subset.txt', 'training.txt', 'validation.txt', 'test_set.txt']
        output_files = ['cleaned_data.txt', 'cleaned_training.txt', 'cleaned_validation.txt', 'cleaned_test_set.txt']
        print('Preprocessing...')
        for input_file, output_file in zip(input_files, output_files):
            print('Preprocessing {0}, cleaned file named {1}'.format(input_file, output_file))
            with open('data/' + output_file, 'w') as outf, codecs.open('data/' + input_file, 'r', 'utf-8') as inf:
                for line in inf:
                    toks = line.strip().split('\t')
                    title = toks[1]
                    toks[1] = remove_low_freq_words(title, counter, 5)
                    outf.write('\t'.join(toks) + '\n')


    def simple_classifier(self):
        def load_data(file):
            X = []
            y = []
            ids = []
            with codecs.open(file, 'r', 'utf-8') as f:
                for line in f:
                    toks = line.strip().split('\t')
                    title = regex.sub('', toks[1])
                    venue = toks[2]
                    X.append(title)
                    y.append(venue)
                    ids.append(toks[0])
            return ids, X, y

        vectorizer = CountVectorizer(lowercase=True, min_df=5)



        print('Fitting label encoder by data/labels.txt...')
        venues_file = 'data/labels.txt'
        with codecs.open(venues_file, 'r', 'utf-8') as f:
            venues = [line.strip() for line in f]
        self.label_encoder.fit(venues)

        print('Fitting vectorizer by cleaned_training.txt (sample features)...')
        train_file = 'data/cleaned_training.txt'
        _, train_X, train_y = load_data(train_file)
        vectorizer.fit_transform(train_X)

        print('Transforming subset data and storing...')
        with codecs.open('output/text_features.txt', 'w', 'utf-8') as outf, codecs.open('data/cleaned_data.txt', 'r', 'utf-8') as inf:
            for line in inf:
                toks = line.strip().split('\t')
                title = toks[1]
                label = toks[2]
                feature_vector = vectorizer.transform([title]).toarray()[0]
                label_enc = self.label_encoder.transform([label])[0]
                outf.write(','.join([str(i) for i in feature_vector]) + '\t' + str(label_enc) + '\n')

        print('Transforming training data...')
        train_X = vectorizer.transform(train_X)
        train_y = self.label_encoder.transform(train_y)

        print('Transforming validation data...')
        valid_file = 'data/cleaned_validation.txt'
        _, valid_X, valid_y = load_data(valid_file)
        valid_X = vectorizer.transform(valid_X)
        valid_y = self.label_encoder.transform(valid_y)

        print('Transforming test data...')
        test_file = 'data/cleaned_test_set.txt'
        ids, test_X, _ = load_data(test_file)
        test_X = vectorizer.transform(test_X)

        print('Fitting Linear Model...')
        clf = linear_model.SGDClassifier(tol=1e-3, max_iter=1000)
        clf.fit(train_X, train_y)

        print('Testing...')
        test_y = clf.predict(test_X)
        pred_res = 'output/text_feature_predictions.txt'
        with open(pred_res, 'w') as f:
            for i, pred in enumerate(test_y):
                id = ids[i]
                venue = self.label_encoder.inverse_transform(pred)
                f.write(id + '\t' + venue + '\n')

        output_file = open('output/result_simple_clf.txt', 'w')
        pred_y = clf.predict(valid_X)
        micro = f1_score(valid_y, pred_y, average='micro')
        macro = f1_score(valid_y, pred_y, average='macro')
        # acc = accuracy_score(valid_y, pred_y)
        output_str = 'f1 score micro: {0}, f1 score macro: {1}'.format(micro, macro)
        print('=========== Simple Classifier Result ===========')
        print(output_str)
        print('================================================')
        output_file.write(output_str)


    def hin_classifier(self):
        def load_data(file):
            X = []
            y = []
            ids = []
            with codecs.open(file, 'r', 'utf-8') as f:
                for line in f:
                    id, title, pvenue, _, cvenue = line.strip().split('\t')
                    title = regex.sub('', title)
                    feature = ' '.join([title, cvenue])
                    X.append(feature)
                    ids.append(id)
            return ids, X, y

        vectorizer = CountVectorizer(lowercase=True, min_df=5)

        print('Fitting vectorizer by cleaned_training.txt (HIN features)...')
        train_file = 'data/cleaned_training.txt'
        _, train_X, train_y = load_data(train_file)
        vectorizer.fit_transform(train_X)



        print('Transforming training data...')
        train_X = vectorizer.transform(train_X)
        train_y = self.label_encoder.transform(train_y)

        print('Transforming validation data...')
        valid_file = 'data/cleaned_validation.txt'
        _, valid_X, valid_y = load_data(valid_file)
        valid_X = vectorizer.transform(valid_X)
        valid_y = self.label_encoder.transform(valid_y)

        print('Transforming test data...')
        test_file = 'data/cleaned_test_set.txt'
        ids, test_X, _ = load_data(test_file)
        test_X = vectorizer.transform(test_X)

        print('Fitting Linear Model...')
        clf = linear_model.SGDClassifier(tol=1e-3, max_iter=1000)
        clf.fit(train_X, train_y)

        print('Testing...')
        test_y = clf.predict(test_X)
        pred_res = 'output/text_feature_predictions.txt'
        with open(pred_res, 'w') as f:
            for i, pred in enumerate(test_y):
                id = ids[i]
                venue = self.label_encoder.inverse_transform(pred)
                f.write(id + '\t' + venue + '\n')

        output_file = open('output/result_hin_clf.txt', 'w')
        pred_y = clf.predict(valid_X)
        micro = f1_score(valid_y, pred_y, average='micro')
        macro = f1_score(valid_y, pred_y, average='macro')
        # acc = accuracy_score(valid_y, pred_y)
        output_str = 'f1 score micro: {0}, f1 score macro: {1}'.format(micro, macro)
        print('=========== Simple Classifier Result ===========')
        print(output_str)
        print('================================================')
        output_file.write(output_str)


def main():
    PVS = PublicationVenueSuggestion()
    # PVS.data_cleaning()
    PVS.simple_classifier()
    # PVS.advanced_classifier()


if __name__ == "__main__":
    main()