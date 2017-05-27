import csv
import numpy as np

class TrainData:
    def __init__(self, train_filename, labels=25000, values=384):
        self.labels = labels
        self.values = values
        self.train_filename = train_filename
        self.value = np.zeros((self.labels, self.values), dtype=float)
        self.reference = np.zeros((self.labels, 1), dtype = float)
        self.read_train_file()

    def read_train_file(self):
        with open(self.train_filename) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                i = int(row[0])
                for k in range(1, self.values):
                    self.value[i, k-1] = float(row[k])
                self.reference[i, 0] = row[self.values+1]

    def get_value(self):
        return self.value

    def get_reference(self):
        return self.reference

class TestData:
    def __init__(self, test_filename, submission_filename, labels=25000, values=384):
        self.labels = labels
        self.values = values
        self.test_filename = test_filename
        self.submission_filename = submission_filename
        self.value = np.zeros((self.labels, self.values), dtype=float)
        #self.reference = numpy.zeros((self.labels, 1), dtype = float)
        self.read_test_file()

    def read_test_file(self):
        with open(self.test_filename) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                i = int(row[0])
                for k in range(1, self.values+1):
                    self.value[i, k-1] = float(row[k])

    def get_value(self):
        return self.value

    def output(self):
        with open(self.submission_filename, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['Id', 'reference'])
            index = 0
            for i in self.reference:
                row = [str(index), str(i[0])]
                f_csv.writerow(row)
