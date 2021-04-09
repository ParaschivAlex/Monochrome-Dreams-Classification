# Pana la linia 77 este totul ca la Naive Bayes

import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing


def load_train_data():
    train_images = []
    for image in glob.glob("./train/*.png"):  # incarc pathul imaginilor cu glob in variabila image, glob returneaza o lista cu path name-urile care se potrivesc in argument. Pathname-ul poate fi absolut dar si relativ.
        train_images.append(plt.imread(image))  # plt.image ia calea data mai sus si citeste imaginea un format binar. Imi imparte valorile pixelilor la 255.
    images = np.array(train_images)  # plt.image deja returna un numpy array dar l-am pus
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


loaded_train_images = load_train_data()  # aici incarc datele si am cateva printuri pentru a vedea daca este totul ok
print(len(loaded_train_images))
loaded_validation_images = load_validation_data()
loaded_test_images = load_test_data()
print(loaded_train_images)

length_train, train_x_axis, train_y_axis = loaded_train_images.shape  # iau shapeul datelor citite pentru a face reshape mai tarziu
length_validation, validation_x_axis, validation_y_axis = loaded_validation_images.shape
length_test, test_x_axis, test_y_axis = loaded_test_images.shape

label_train = []
train_name_files = []
f = open("train.txt", "r")  # citesc din cele 3 fisiere denumirea imaginilor si labelul (unde este cazul)
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
for line in f:
    test_name_files.append(line[0:10])  # pana la 10 ca sa evit operatorul new line
f.close()

load_reshaped_train = loaded_train_images.reshape(length_train, train_x_axis * train_y_axis)  # reshapeuiesc datele din vector 3-dimensional intr-unul 2-dimensional
load_reshaped_validation = loaded_validation_images.reshape(length_validation, validation_x_axis * validation_y_axis)  # cu formula arr.respahe(nr_imgs, linie * coloana)
load_reshaped_test = loaded_test_images.reshape(length_test, test_x_axis * test_y_axis)


def compute_accuracy(gt_labels, predicted_labels):  # am definit o functia ca in laborator care sa imi calculeze scorul predictiilor
    accuracy = np.sum(predicted_labels == gt_labels) / len(predicted_labels)  # scorul este numarul de labeluri corecte pe numarul tot de labeluri lol :)
    return accuracy


svm_model = svm.SVC(C=4, kernel='rbf')  # definesc un model SVM cu C=4, kernel = 'rbf', gamma default scale
svm_model.fit(load_reshaped_train, label_train)  # antrenez modelul ca in laborator
predicted_labels_svm = svm_model.predict(load_reshaped_validation)  # dau predict pe validari
model_accuracy_svm = compute_accuracy(np.asarray(label_validation), predicted_labels_svm)  # vad ce scor a dat pe datele de validare

print('SVM model accuracy for C=%d is %f' % (4, model_accuracy_svm))  # am testat modelul pentru C=1,21 kernel rbf, si pentru linear cateva valori


# cele mai bune rezultate au fost pe C=4 si C=9, pentru C >= 12 deja a putut fi observat un overfitting atunci cand am dat submit pe kaggle

def confusion_matrix(label_true, label_predicted):  # aici afisez matricea de confuzie
    num_classes = max(max(label_true), max(label_predicted)) + 1  # iau numarul de clase posibile, puteam sa ii dau 9 eu
    conf_matrix = np.zeros((num_classes, num_classes))  # face o matrice 9x9 initializata cu 0

    for i in range(len(label_true)):  # iau i = numarul de labeluri date (de imagini)
        conf_matrix[int(label_true[i]), int(label_predicted[i])] += 1  # daca prezicerea este corecta crestem valoarea pe diagonala principala, daca nu in afara ei ( [i][j] i -> ce trebuia prezis si j-> ce a prezis)
    return conf_matrix


print(confusion_matrix(label_validation, predicted_labels_svm))

g = open("sample_submission.txt", "w")
sample_sub = svm_model.predict(load_reshaped_test)  # aici dau predict pe datele de test
print(len(sample_sub), len(test_name_files))  # dau print la lungime sa vad daca a dat predict la toate datele
g.write("id,label\n")
for i in range(len(sample_sub)):
    g.write(str(test_name_files[i]) + "," + str(sample_sub[i]) + '\n')  # Scrierea in fisier
g.close()
