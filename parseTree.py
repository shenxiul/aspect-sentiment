import nltk
import nltk.parse.stanford
import csv
import os
import pdb
from unidecode import unidecode
import json

#os.environ['JAVA_HOME'] = 'C://Program Files//Java//jdk1.8.0_45//bin'
#os.environ['STANFORD_PARSER'] = 'D://stanford-parser-full-2015-04-20'
#os.environ['STANFORD_MODELS'] = 'D://stanford-parser-full-2015-04-20'

os.environ['STANFORD_PARSER'] = '../stanford-parser-full-2015-04-20'
os.environ['STANFORD_MODELS'] = '../stanford-parser-full-2015-04-20'


class Node(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.left = None
        self.right = None
        self.word = None
        self.is_leaf = False


def parse_sent(parser, sent_list):
    sentences = parser.parse_sents(sent_list)
    return sentences


def load(filename):
    input_file = open(filename, 'r')
    csv_reader = csv.reader(input_file)
    data_set = {}
    for line in csv_reader:
        review = unidecode(line[0].decode('utf-8'))
        if not all(ord(c) < 128 for c in review): pdb.set_trace()
        if review not in data_set:
            data_set[review] = [{'aspect': line[1], 'rating': float(line[2])}]
        else:
            data_set[review].append({'aspect': line[1], 'rating': float(line[2])})
    input_file.close()
    return data_set


def tokenize(data_set):
    index = 0
    tok_review_set = []
    review_set = []
    for review in data_set:
        data_set[review].append(index)
        index += 1
        tok_review_set.append(nltk.word_tokenize(review))
        review_set.append(review)
    return tok_review_set, review_set


def binarize(tree):
    nltk.tree.Tree.collapse_unary(tree, True, True)
    nltk.tree.Tree.chomsky_normal_form(tree)
    return tree

if __name__ == '__main__':
    parser = nltk.parse.stanford.StanfordParser()
    data_set = load('./data/final_review_set.csv')
    token_set, review_set = tokenize(data_set)
    max_number = len(data_set)
    forest = parse_sent(parser, token_set[:max_number])
    sents = []
    for tree in forest:
        sents.append(list(tree)[0])
    sents = [binarize(sent) for sent in sents]
    data = []
    for parsed_tree, review in zip(sents, review_set[:max_number]):
        data.append({"tree": str(parsed_tree), "label": data_set[review][:-1]})
    train_sample = int(len(data_set) * 0.7)
    dev_sample = int(len(data_set) * 0.1)
    test_sample = len(data_set) - train_sample - dev_sample
    data_idx = list(range(len(data_set)))
    train_data = data[:train_sample]
    dev_data = data[train_sample:train_sample+dev_sample]
    test_data = data[train_sample+dev_sample:]
    with open('./data/train.json', 'w') as train_file: json.dump(train_data, train_file)
    with open('./data/dev.json', 'w') as dev_file: json.dump(dev_data, dev_file)
    with open('./data/test.json', 'w') as test_file: json.dump(test_data, test_file)
