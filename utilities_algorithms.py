import random
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.cluster import KMeans
from scipy.spatial import distance


# Return the cluster center from kmeans algorithm
def kmeans(k, list_descriptor):
    k_means = KMeans(n_clusters=k, n_init=10)
    k_means.fit(list_descriptor)
    visual_words = k_means.cluster_centers_
    return visual_words


def plot_patches(patch_list: list):
    count = len(patch_list)  # number of patch
    n_rows = int(np.ceil((count // 10 + 1) * 1.5))  # at least 2 row
    plt.figure()
    for patch_id, i in enumerate(patch_list):
        plt.subplot(n_rows, int(np.ceil(count / n_rows)), patch_id + 1)
        plt.imshow(i)


# input a key_point and image, then calculate the region that the key_point describe
def cut_img(kp, image):
    radius = int(np.round(kp['size']))
    pt = kp['pt']
    x = int(round(pt[0]))
    y = int(round(pt[1]))
    left = max(0, x - radius)
    high = min(len(image[0]), y + radius)
    right = min(len(image), x + radius)
    low = max(0, y - radius)
    return image[left: right, low: high]


def visualize_patches(descriptors, keypoint_vectors, centers, visualize_number, dataset):
    """
        visualize_number: an int between 0 to 20, visualize a number between 0 to 20 image patches in same code word.
        :return a list of image patches that are assigned to the same codeword.
    """
    patch_list = []

    if visualize_number > 20:
        visualize_number = 20
    # Random Pick a center
    sample_center = random.choice(centers)
    visual_count = 0
    for img_class, descriptor_value in descriptors.items():
        for img_idx, image in enumerate(descriptor_value):
            # similar to find_index
            for feature_idx, each_feature in enumerate(image):
                count = 0
                ind = 0
                for i in range(len(centers)):
                    if i == 0:
                        count = distance.euclidean(each_feature, centers[i])
                    else:
                        dist = distance.euclidean(each_feature, centers[i])
                        if dist < count:
                            count = dist
                            ind = i
                # check if it is equal to sample center
                if (centers[ind] == sample_center).all():
                    # get keypoint by mapping the index of descriptors
                    key_point = keypoint_vectors[img_class][img_idx][feature_idx]
                    # get img from dataset by mapping the index
                    image = dataset[img_class][img_idx]
                    img_show = cut_img(key_point, image)
                    patch_list.append(img_show)
                    visual_count += 1
                if visual_count >= visualize_number: return patch_list
    return patch_list


def plot_correct(patch_list: list, title):
    count = len(patch_list)  # number of patch
    n_rows = int(np.ceil((count // 10 + 1) * 1.5))  # at least 2 row
    plt.figure()
    plt.suptitle(title, fontsize=10)
    for patch_id, i in enumerate(patch_list):
        plt.subplot(n_rows, int(np.ceil(count / n_rows)), patch_id + 1)
        plt.imshow(i)
        plt.axis('off')
    plt.show()


def show_classified(dataset, correct, incorrect):
    all_pic = []

    for k in dataset.keys():
        for i in dataset[k]:
            all_pic.append(i)

    for k in list(dataset.keys()):
        p_correct = correct[k]
        p_incorrect = incorrect[k]
        correct_pic = []
        incorrect_pic = []
        for c in p_correct:
            correct_pic.append(all_pic[c])
        plot_correct(correct_pic, f"Correctly classified {k}")

        for i_c in p_incorrect:
            incorrect_pic.append(all_pic[i_c])
        plot_correct(incorrect_pic, f"Incorrectly classified {k}")


def draw_confusion_matrix(cm, title, classes):
    # Confusion matrix graph
    plt.figure(figsize=(14, 14))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=30)
    plt.colorbar(fraction=0.045)
    tick_marks = np.arange(len(classes))
    plt.ylim(-0.5, 4.5)
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    plt.gca().invert_yaxis()

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.show()


def get_xy(histo):
    x = []
    y = []
    n = 0

    for image_class, images in histo.items():
        n += 1
        for image in images:
            x.append(image)
            y.append(image_class)
    return [x, y]


def divide_into_correct_and_incorrect(key, test, predict_y):
    correct = {}
    incorrect = {}
    for i in range(len(key)):
        class_correct = []
        class_incorrect = []
        for j in range(len(predict_y)):
            if predict_y[j] == key[i] or test[j] == key[i]:
                if test[j] == predict_y[j]:
                    class_correct.append(j)
                else:
                    if test[j] == key[i]:
                        class_incorrect.append(j)
        correct[key[i]] = class_correct
        incorrect[key[i]] = class_incorrect
    return correct, incorrect
