from numpy import *
from scipy.special import expit
import time

def loadDataSet(Path):
    dataMat = [];
    labelMat = [];
    with open(Path) as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            labelMat.append(int(lineArr[0]))
            del(lineArr[0])
            temp = [1.0]
            for i in range(1, 202):
                temp.append(0.0)
            for word in lineArr:
                word = word.split(":")
                temp[int(word[0])] = float(word[1])
            dataMat.append(temp)
    print("Successfully load from "+ Path)
    return dataMat, labelMat

def sigmoid(inX):
    return expit(inX)

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        localtime = time.asctime(time.localtime(time.time()))
        print("Round:"+str(j+1)+ " "+ localtime)
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(i+j+1.0)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def testOutput(weights, testArr, answer):
    weights = mat(weights).transpose()
    output = testArr*weights
    m, n = shape(output)
    with open(answer, "w") as f:
        f.write("id,label\n")
        for i in range(m):
            temp = round(sigmoid(output[i,0]),2)
            print(temp)
            f.write(str(i)+","+str(temp)+"\n")

if __name__ == "__main__":
    #trainPath
    #testPath
    #ansPath
    dataArr, labelMat = loadDataSet("../data/train_data.txt")
    weights = stocGradAscent1(array(dataArr), labelMat, 1)
    #weights=ones(202)
    testArr, tId = loadDataSet("../data/test_data.txt")
    testOutput(weights, mat(testArr), "answer.txt")

