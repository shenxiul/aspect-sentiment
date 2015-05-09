import loadFile
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

if __name__ == '__main__':
    transformer = TfidfTransformer()
    vectorizer = CountVectorizer(min_df=1)
    data = loadFile.load('./data/final_review_set.csv')
    reviews = [each_data['review'] for each_data in data]
    X = vectorizer.fit_transform(reviews)
    tfidf = transformer.fit_transform(X)

    aspect_label = collect_aspect_label(data)
    rating_label = collect_rating_label(data)

    # tokenize(data)
    # reviews_tokenized = [each_data['review'] for each_data in data]
    # voc = build_vocabulary(reviews_tokenized)
