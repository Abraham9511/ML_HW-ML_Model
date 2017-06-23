import numpy as np
import multiprocessing
#import threading
from time import ctime
import read_data

train_filename = "../data/save_train.csv"
test_filename = "../data/save_test.csv"
submission_filename = "submission.csv"


def profile(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
    return wrapper
    #def __init__(self, func, args):
    #    threading.Thread.__init__(self)
    #    self.func = func
    #    self.args = args
    #    ```

    #def run(self):
    #    apply(self.func, self.args)
        #self.func(self.args)

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

@profile
def predict(stage):
    t1 = multiprocessing.Process(target=train,args=(0,stage, theta1,'Thread 1', ))
    t2 = multiprocessing.Process(target=train,args=(stage+1,stage*2+1,theta2,'Thread 2', ))
    t3 = multiprocessing.Process(target=train,args=(stage*2+2,stage*3+2,theta3,'Thread 3', ))
    t4 = multiprocessing.Process(target=train,args=(stage*3+3,len(train_value)-1,theta4, 'Thread 4'))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

if __name__ == '__main__':
    trainData = read_data.TrainData(train_filename)
    testData = read_data.TestData(test_filename, submission_filename)
    print("Load Data Finish, "+ ctime())
    train_value = trainData.get_value()
    test_value = testData.get_value()
    theta = np.random.normal(0,1,len(train_value[0]))
    theta.shape = (len(train_value[0]-1), 1)
    theta = np.transpose(theta)
    theta1 = theta
    theta2 = theta
    theta3 = theta
    theta4 = theta
    stage = len(train_value)//4
    print('stage: '+ str(stage))
    print('train_value'+str(train_value.shape))
    print('test_value'+ str(test_value.shape))

    predict(stage)

    print('theta finish')
    theta = (theta1 + theta2 + theta3 +theta4)/4
    print(theta)
    testData.output(np.dot(test_value, theta.transpose()))
    #train(trainData, testData)

