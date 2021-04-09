import numpy as np
import glob
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


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
# print(len(loaded_train_images))
loaded_validation_images = load_validation_data()
loaded_test_images = load_test_data()
# print(train_image[0])

# plt.imshow(loaded_train_images[0]) # voiam sa afisez prima imagine pentru a vedea daca este ok
# plt.show()

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


def values_to_bins(x, bins):  # functie ca in laborator care primeste o matrice 2d si capetele intervalelor de la linia 89
    x = np.digitize(x, bins)  # calculez indexul intervalului corespunzator (discretizez multimea de antrenare, validare si testare)
    return x - 1


load_reshaped_train = loaded_train_images.reshape(length_train, train_x_axis * train_y_axis)  # reshapeuiesc datele din vector 3-dimensional intr-unul 2-dimensional
load_reshaped_validation = loaded_validation_images.reshape(length_validation, validation_x_axis * validation_y_axis)  # cu formula arr.respahe(nr_imgs, linie * coloana)
load_reshaped_test = loaded_test_images.reshape(length_test, test_x_axis * test_y_axis)

# for num_bins in range(3, 15, 1): # aici am testat modelul pe num_bins de la 3 la 14
#     bins = np.linspace(0, 1, num=num_bins)
#     x_train = values_to_bins(load_reshaped_train, bins)
#     x_test = values_to_bins(load_reshaped_validation, bins)
#
#     clf = MultinomialNB()
#     clf.fit(x_train, label_train)
#     print('Accuracy for num_bins=%d is %f' % (num_bins, clf.score(x_test, label_validation)))

num_bins = 12  # am luat cea mai buna valoare si am pregatit modelul iar
bins = np.linspace(0, 1, num=num_bins)  # returneaza num_bins numere distantate uniform pe intervalul [0,1)
x_train = values_to_bins(load_reshaped_train, bins)  # aplic functia digitize
x_validation = values_to_bins(load_reshaped_validation, bins)
x_test = values_to_bins(load_reshaped_test, bins)

clf = MultinomialNB()  # definesc un model multinomialdb
clf.fit(x_train, label_train)  # antrenez datele
print('accuracy =', clf.score(x_validation, label_validation))  # afisez scorul pe datele de validare

predictions = clf.predict(load_reshaped_validation)  # afisez predictiile pe datele de validare si le si retin
print(predictions)


def confusion_matrix(label_true, label_predicted):  # aici afisez matricea de confuzie
    num_classes = max(max(label_true), max(label_predicted)) + 1 #iau numarul de clase posibile, puteam sa ii dau 9 eu
    conf_matrix = np.zeros((num_classes, num_classes)) #face o matrice 9x9 initializata cu 0

    for i in range(len(label_true)): # iau i = numarul de labeluri date (de imagini)
        conf_matrix[int(label_true[i]), int(label_predicted[i])] += 1 #daca prezicerea este corecta crestem valoarea pe diagonala principala, daca nu in afara ei ( [i][j] i -> ce trebuia prezis si j-> ce a prezis)
    return conf_matrix


print(confusion_matrix(label_validation, predictions)) #pe exemplul dat observam ca majoritatea sunt incadrate bine doar label 0 are incadrari proaste

# am comentat scrierea in fisier ca sa nu se ruleze cand mai testam modelul si aveam alte date mai bune (lucram pe acelasi fisier cu toate sursele)
# g = open("sample_submission.txt", "w")
# sample_sub = clf.predict(x_test) #dau predict pe test
# print(len(sample_sub), len(test_name_files)) #verific daca am aceeasi lungime adica daca am dat predict pe toate
# g.write("id,label\n")
# for i in range (len(sample_sub)):
#     g.write(str(test_name_files[i]) + "," + str(sample_sub[i]) + '\n')
#
# g.close()
