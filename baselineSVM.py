from sklearn.svm import LinearSVC
import validation as vld
import loadFile
import numpy as np

reg = 0.1


def single_train(data, label):
    clf = LinearSVC(C=reg)
    return clf.fit(data, label)


def train_both(data, label):
    clf1 = LinearSVC(C=reg)
    clf2 = LinearSVC(C=reg)
    return clf1.fit(data, label[:, 0]), clf2.fit(data, label[:, 1])

if __name__ == '__main__':
    total_data = loadFile.file2mat('./data/final_review_set.csv')
    shuffled_data = vld.data_reshuffle(total_data)
    train_mat = shuffled_data[0]
    aspect_label = shuffled_data[1]
    rating_label = shuffled_data[2]
    label_mat = np.vstack((aspect_label, rating_label)).T
    single_label = aspect_label * len(loadFile.aspect_dic) + rating_label
    print vld.cross_validation(train_mat, aspect_label, single_train, vld.test_single)
    print vld.cross_validation(train_mat, rating_label, single_train, vld.test_single)
    print vld.cross_validation(train_mat, single_label, single_train, vld.test_single)
    print vld.cross_validation(train_mat, single_label, single_train, vld.test_aspect)
    print vld.cross_validation(train_mat, single_label, single_train, vld.test_rating)
    print vld.cross_validation(train_mat, label_mat, train_both, vld.test_mat)
