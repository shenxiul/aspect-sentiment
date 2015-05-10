import csv
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


aspect_dic = {"overall":    0,
              "appearance": 1,
              "taste":      2,
              "palate":     3,
              "aroma":      4}


def tokenize(data_set):
    for i in xrange(len(data_set)):
        data_set[i]['review'] = nltk.word_tokenize(data_set[i]['review'])
    return


def build_vocabulary(review_set):
    vocabulary = {}
    index = 0
    for review in review_set:
        review = set(review)
        for word in review:
            if word not in vocabulary:
                vocabulary[word] = {"index": index, "df": 1}
                index += 1
            else:
                vocabulary[word]["df"] += 1
    return vocabulary


def collect_aspect_label(data_set):
    label = np.zeros(len(data_set), dtype='int32')
    for i, data_entry in enumerate(data_set):
        label[i] = aspect_dic[data_entry['aspect']]
    return label


def collect_rating_label(data_set):
    label = np.zeros(len(data_set), dtype='int32')
    for i, data_entry in enumerate(data_set):
        label[i] = int(data_entry['rating']) - 1
    return label


def load(filename):
    input_file = open(filename, 'r')
    csv_reader = csv.reader(input_file)
    data_set = []
    for line in csv_reader:
        data_set.append({'review': line[0].lower(), 'aspect': line[1], 'rating': float(line[2])})
    return data_set


def file2mat(filename):
    transformer = TfidfTransformer()
    vectorizer = CountVectorizer(min_df=1)
    data = load(filename)
    reviews = [each_data['review'] for each_data in data]
    bag_of_word = vectorizer.fit_transform(reviews)
    tfidf = transformer.fit_transform(bag_of_word)

    aspect_label = collect_aspect_label(data)
    rating_label = collect_rating_label(data)
    return tfidf, aspect_label, rating_label


def load_wordvec(filename):
    wordvec_dic = {}
    with open(filename, 'r') as wordvec_file:
        for i, line in enumerate(wordvec_file):
            content = line.split()
            wordvec_dic[content[0]] = np.array(map(float, content[1:]))
            if i % 10000 == 0:
                print i
    return wordvec_dic


def bag_of_wordvec(tokens, wordvec_dic, dimension):
    vec = np.zeros(dimension)
    for token in tokens:
        try:
            vec += wordvec_dic[token]
        except KeyError:
            pass
            # print "No token %s" % token
    return vec


def file2mat_bag_of_wordvec(filename):
    dimension = 300
    wordvec_dic = load_wordvec('./data/glove.6B.300d.txt')
    data = load(filename)
    tokenize(data)
    reviews = [each_data['review'] for each_data in data]
    bag_of_wordvec_mat = np.zeros((len(reviews), dimension))
    for i in xrange(bag_of_wordvec_mat.shape[0]):
        bag_of_wordvec_mat[i] = bag_of_wordvec(reviews[i], wordvec_dic, dimension)
    aspect_label = collect_aspect_label(data)
    rating_label = collect_rating_label(data)
    return bag_of_wordvec_mat, aspect_label, rating_label


if __name__ == '__main__':
    print file2mat_bag_of_wordvec('./data/final_review_set.csv')
    pass

