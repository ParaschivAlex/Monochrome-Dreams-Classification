#Aici este citirea exact ca la svm si la naive bayes doar ca testez parametri diferite pentru mlpclassifier.
#Rezultatul nu a trecut de 0.72 din ce imi amintesc (stiu ca era mai slaba solutia decat svm si nu am notat-o).

#Am luat MLPCLASSIFIER-ul din laborator si m-am jucat cu valorile ce erau oferite in documentatia din laboratul 7.
import numpy as np
import glob
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def load_train_data():
    train_images = []
    for image in glob.glob("./train/*.png"):
        train_images.append(plt.imread(image))
    images = np.array(train_images)
    return images


def load_validation_data():
    validation_images = []
    for image in glob.glob("./validation/*.png"):
        validation_images.append(plt.imread(image))
    images = np.array(validation_images)
    return images


def load_test_data():
    test_images = []
    for image in glob.glob("./test/*.png"):
        test_images.append(plt.imread(image))
    images = np.array(test_images)
    return images

loaded_train_images = load_train_data()
print(len(loaded_train_images))
loaded_validation_images = load_validation_data()
loaded_test_images = load_test_data()
# print(train_image[0])

length_train, train_x_axis, train_y_axis = loaded_train_images.shape
length_validation, validation_x_axis, validation_y_axis = loaded_validation_images.shape
length_test, test_x_axis, test_y_axis = loaded_test_images.shape

label_train = []
train_name_files = []
f = open("train.txt", "r")
for line in f:
    separator = line.split(",")
    train_name_files.append(separator[0])
    label_train.append(int(separator[1]))
f.close()
print(len(label_train))

label_validation = []
validation_name_files = []
f = open("validation.txt", "r")
for line in f:
    separator = line.split(",")
    validation_name_files.append(separator[0])
    label_validation.append(int(separator[1]))
f.close()

test_name_files = []
f = open("test.txt", "r")
#ignore_first_line = f.readline()
for line in f:
    test_name_files.append(line[0:10])
f.close()

load_reshaped_train = loaded_train_images.reshape(length_train, train_x_axis*train_y_axis)
load_reshaped_validation = loaded_validation_images.reshape(length_validation, validation_x_axis*validation_y_axis)
load_reshaped_test = loaded_test_images.reshape(length_test, test_x_axis*test_y_axis)

def train_and_eval(clf):
    clf.fit(load_reshaped_train, label_train)
    return clf.score(load_reshaped_validation, label_validation)

#Aici am doar cateva din modele testate

# clf = MLPClassifier(hidden_layer_sizes=(1), activation='tanh',
#                     learning_rate_init=0.01, momentum=0)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(10), activation='tanh',
#                     learning_rate_init=0.01, momentum=0)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu',
#                     learning_rate_init=0.01, momentum=0,
#                     max_iter=2000)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu',
#                     learning_rate_init=0.01, momentum=0.9,
#                     max_iter=2000)
# print(train_and_eval(clf))
#
#
# clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu',
#                     learning_rate_init=0.01, momentum=0.9,
#                     max_iter=2000, alpha=0.005)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(30, 30,30), activation='relu',
#                     learning_rate_init=0.05,
#                     max_iter=2000, alpha=0.005)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu',
#                     learning_rate_init=0.001,
#                     max_iter=2000, alpha=0.005)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(50,50), activation='relu',
#                     learning_rate_init=0.1,
#                     max_iter=2000, alpha=0.005)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(10,10,10), activation='relu',
#                     learning_rate_init=0.05,
#                     max_iter=2000, alpha=0.005)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(50,100,50), activation='relu',
#                     learning_rate_init=0.001,
#                     max_iter=2000, alpha=0.005)
# print(train_and_eval(clf))
#
# clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
#                     learning_rate_init=0.1,
#                     max_iter=2000, alpha=0.005)
# print(train_and_eval(clf))

# 0.1108
# 0.451
# 0.3454
# 0.4824
# 0.4986
# 0.1066
# 0.6682
# 0.1122
# 0.2158
# 0.704
# 0.104

#Fac matricea de confuzie pentru penultimul model



clf = MLPClassifier(hidden_layer_sizes=(50,100,50), activation='relu',
                    learning_rate_init=0.001, #rata de invatare este cea default
                    max_iter=2000, alpha=0.005) #numarul maxim de epoci si parametrul pentru regularizarea L2


print("Accuracy =", train_and_eval(clf))
predictions = clf.predict(load_reshaped_validation)

def confusion_matrix(label_true, label_predicted):  # aici afisez matricea de confuzie
    num_classes = max(max(label_true), max(label_predicted)) + 1  # iau numarul de clase posibile, puteam sa ii dau 9 eu
    conf_matrix = np.zeros((num_classes, num_classes))  # face o matrice 9x9 initializata cu 0

    for i in range(len(label_true)):  # iau i = numarul de labeluri date (de imagini)
        conf_matrix[int(label_true[i]), int(label_predicted[i])] += 1  # daca prezicerea este corecta crestem valoarea pe diagonala principala, daca nu in afara ei ( [i][j] i -> ce trebuia prezis si j-> ce a prezis)
    return conf_matrix


print(confusion_matrix(label_validation, predictions))

# g = open("sample_submission.txt", "w")
# sample_sub = clf.predict(load_reshaped_test)
# print(len(sample_sub), len(test_name_files))
# g.write("id,label\n")
# for i in range (len(sample_sub)):
#     g.write(str(test_name_files[i]) + "," + str(sample_sub[i]) + '\n')
#
# g.close()