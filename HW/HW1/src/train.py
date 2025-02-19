import numpy as np
from time import ctime
import read_data

train_filename = "../data/save_train.csv"
test_filename = "../data/save_test.csv"
submission_filename = "submission.csv"

def train(start, finish, theta, name):
    global trainData
    global testData
    train_value = trainData.get_value()[start:finish,]
    test_value = testData.get_value()[start:finish,]
    train_reference = trainData.get_reference()[start:finish,]
    # test_reference = np.zeros((testData.labels, 1), dtype=float)

    m = len(train_value)
    n = len(train_value[0])
    print('m: '+str(m))
    print('n: '+str(n))
    alpha = 0.05
    eps = 0.0001
    max_iter = 5000
    miter = 0
    flag = 0

    while True:
        for i in range(m):
            deviation = 0
            h = np.dot(theta * alpha, train_value[i])
            err = train_reference[i][0]-h
            err = err*alpha
            for j in range(n):
                theta[0][j] = theta[0][j] + err * train_value[i][j]

        miter = miter +1

        for i in range(m):
            deviation = deviation + (train_reference[i][0]- np.dot(theta, train_value[i]))**2

        print(ctime()+ " "+ name + " deviation: "+ str(deviation))
        if deviation < eps or miter > max_iter:
            flag = 1
            break

        if flag == 1:
            break
    print(theta)
#    test_reference = np.dot(test_value, theta)
#    testData.output(test_reference)

if __name__ == '__main__':
    trainData = read_data.TrainData(train_filename)
    testData = read_data.TestData(test_filename, submission_filename)
    print("Load Data Finish, "+ ctime())
    train_value = trainData.get_value()
    test_value = testData.get_value()
    theta = np.random.normal(0,1,len(train_value[0]))
    theta.shape = (len(train_value[0]-1), 1)
    theta = np.transpose(theta)
    print('train_value'+str(train_value.shape))
    print('test_value'+ str(test_value.shape))


    train(0, len(train_value)-1, theta)

    print('theta finish')
    print(theta)
    testData.output(np.dot(test_value, theta))

