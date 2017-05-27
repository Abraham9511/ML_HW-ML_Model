import read_data
import xgboost as xgb
import csv

train_filename = "../data/train_data.txt"
test_filename = "../data/test_data.txt"
submission = "./../result/submission.csv"

def train(trainData, testData):
    dTrain = xgb.DMatrix(trainData.get_features(), trainData.get_labels())
    dTest = xgb.DMatrix(testData.get_features())

    param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
    param['nthread'] = 4
    plst = param.items()
    plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way
    plst += [('eval_metric', 'ams@0')]
    num_round = 2

    bst = xgb.train(param, dTrain, num_round)

    test_labels = bst.predict(dTest)

    print(test_labels)

    with open(submission, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['id', 'label'])
        for index in range(0, 282796):
            f_csv.writerow([str(index), str(test_labels[index])])

if __name__ == '__main__':
    trainData = read_data.Data(train_filename)
    trainData.load_all_data()
    testData = read_data.Data(test_filename)
    testData.load_all_data()
    train(trainData, testData)


