import cv2
import numpy as np

from scipy.spatial import distance


def to_inter(hist1, hist2):
    return cv2.compareHist(np.float32(hist1), np.float32(hist2), cv2.HISTCMP_INTERSECT)


class KnnClassifier:
    # initialize with k = 4
    def __init__(self, k=4, dis_type='L2'):
        self.key = k
        self.dis_type = dis_type
        self.xtr, self.ytr = None, None

    def fit(self, xtr, ytr):
        self.xtr = xtr
        self.ytr = ytr

    # to predict the digits
    def predict(self, x_test1):
        num = len(x_test1)
        y_predict = list(range(len(x_test1)))
        for i in range(num):
            # Calculate the distance based on p, p = 1 Manhattan, p = 2 euclidean distance
            distances = []
            for image_class_, z in enumerate(self.xtr):
                if self.dis_type != 'L2':
                    d = to_inter(z, x_test1[i])
                else:
                    d = distance.euclidean(z, x_test1[i])
                distances.append(d)
            distances = np.array(distances)
            if self.dis_type == 'L2':
                # Sort and get the index of first key distance
                sorted_distances = np.argsort(distances)[:self.key]
            if self.dis_type == 'inter':
                sorted_distances = np.argsort(distances)[::-1]
                sorted_distances = sorted_distances[:self.key]
            # Accumulate the count for each possible prediction label
            accumulate = np.zeros(len(np.unique(self.ytr)))
            classes = np.unique(self.ytr)
            for z in sorted_distances:
                # ytr[z] = the prediction result for one result corresponding k_value
                class_index = np.where(classes == self.ytr[z])[0][0]
                accumulate[class_index] += 1
            # Check the index of the most possible prediction, index = the prediction result
            y_predict[i] = classes[np.argmax(accumulate)]
        return np.array(y_predict)


