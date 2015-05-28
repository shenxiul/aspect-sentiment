import nltk
import csv
import os

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


def parse_sent(parser, sent):
    sentence = list(parser.parse(sent))[0]
    nltk.tree.Tree.collapse_unary(sentence, True, True)
    nltk.tree.Tree.chomsky_normal_form(sentence)
    return sentence


def load_sas(filename):
    input_file = open(filename, 'r')
    csv_reader = csv.reader(input_file)
    data_set = []
    for line in csv_reader:
        data_set.append({'review': line[0].lower(), 'aspect': line[1], 'rating': float(line[2])})
    return data_set


def tokenize(data_set):
    for i in xrange(len(data_set)):
        data_set[i]['review'] = nltk.word_tokenize(data_set[i]['review'])
    return

if __name__ == '__main__':
    parser = nltk.parse.stanford.StanfordParser()
    data_set = load_sas('./data/final_review_set.csv')
    tokenize(data_set)
    sentence = data_set[0]['review']
    tree = parse_sent(parser, sentence)
