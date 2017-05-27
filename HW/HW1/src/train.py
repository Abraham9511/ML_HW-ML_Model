import numpy as np
import read_data

train_filename = "../data/save_train.csv"
test_filename = "../data/save_test.csv"
submission_filename = "submission.csv"

def train(trainData, testData):
    train_value = trainData.get_value()
    test_value = testData.get_value()
    train_reference = trainData.get_reference()
    test_reference = np.zeros((testData.labels, 1), dtype=float)
    l_index = 0
    for label in test_value:
        min_value = 99999
        m_reference = 0
        index = 0
        for tLine in train_value:
            temp = np.linalg.norm(tLine-label)
            if (min_value > temp):
                min_value = temp
                m_reference = train_reference[index, 0]
            index = index+1
        test_reference[l_index, 0] = m_reference
        l_index = l_index+1
        print("Compute label "+ str(l_index) + ", reference is "+ str(m_reference))
    testData.output()

if __name__ == '__main__':
    trainData = read_data.TrainData(train_filename)
    testData = read_data.TestData(test_filename, submission_filename)
    train(trainData, testData)

