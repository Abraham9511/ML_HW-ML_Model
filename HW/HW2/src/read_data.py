import numpy as np

class Data:
    def __init__(self, filename, case=1866819, fea = 132):
        self.filename = filename
        self.case = case
        self.fea = fea
        self.features = np.zeros((case, fea))
        self.labels = np.zeros((case, 1))

    def process_line(self, line):
        data_line = str(line).split()
        label = int(data_line[0])
        keys = []
        values = []
        for i in range(1, len(data_line)):
            key_value = data_line[i].split(":")
            key = int(key_value[0])
            if key > 132:
                continue
            value = float(key_value[1])
            keys.append(key)
            values.append(value)

        train_data_line = [0] * 132
        for i in range(1, 133):
            try:
                tmp = keys.index(i)
                train_data_line[i-1] = values[tmp]
            except ValueError:
                pass
        feature = np.array(train_data_line)
        feature.shape = (1,132)
        label = np.array(label)
        label.shape = (1,1)
        return feature, label

    def read_data(self):
        with open(self.filename) as f:
            for line in f:
                feature, label = self.process_line(line)
                yield feature, label

    def load_all_data(self):
        index = 0
        for i in self.read_data():
            self.features[index] = i[0][0]
            self.labels[index] = i[1][0]
            index = index +1

    def get_labels(self):
        return self.labels

    def get_features(self):
        return self.features
