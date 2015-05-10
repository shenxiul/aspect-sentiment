
def cross_validation(data_train, data_label, train_method, validate_method, fold=10):
    instance = data_train.shape[0]
    test_num = instance / 10
    accuracy = 0
    for fold_idx in xrange(fold):
        test_idx = set(range(fold_idx * test_num, min((fold_idx + 1) * test_num, instance)))
        train_idx = [i for i in range(instance) if i not in test_idx]
        test_idx = list(test_idx)
        model = train_method(data_train[train_idx], data_label[train_idx])
        accuracy_curr = validate_method(data_train[test_idx], data_label[test_idx], model)
        accuracy += accuracy_curr
    accuracy /= fold
    return accuracy


