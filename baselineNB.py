import loadFile
from sklearn.naive_bayes import MultinomialNB
import random
import numpy as np
from validation import cross_validation


def naive_bayes_single_train(data, label):
    clf = MultinomialNB()
    return clf.fit(data, label)


def naive_bayes_single_test(data, label, model):
    prediction = model.predict(data)
    return float((prediction == label).sum()) / len(label)


def data_reshuffle(data_list):
    data_len = data_list[0].shape[0]
    index = range(data_len)
    random.shuffle(index)
    return [data[index] for data in data_list]

if __name__ == '__main__':
    total_data = loadFile.file2mat('./data/final_review_set.csv')
    shuffled_data = data_reshuffle(total_data)
    train_mat = shuffled_data[0]
    aspect_label = shuffled_data[1]
    rating_label = shuffled_data[2]
    label_mat = np.vstack((aspect_label, rating_label)).T
    single_label = aspect_label * len(loadFile.aspect_dic) + rating_label
    print cross_validation(train_mat, aspect_label, naive_bayes_single_train, naive_bayes_single_test)
    print cross_validation(train_mat, rating_label, naive_bayes_single_train, naive_bayes_single_test)
    print cross_validation(train_mat, single_label, naive_bayes_single_train, naive_bayes_single_test)
