from src.layers.convolution import Conv, FastConv
from src.layers.dense import Dense
from src.layers.flatten import Flatten
from src.layers.pool import MaxPool
from src.models import sequential
from src.optimizers import sgd, momentum, adam
from tensorflow.keras import datasets
from src.losses import categorical_crossentropy as crossentropy, l2
import matplotlib.pyplot as plt
import time
import numpy as np

np.seterr(divide="ignore")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

classifier = sequential.SequentialModel()
classifier.add(FastConv(32, (3, 3), input_shape=(32, 32, 3)))
classifier.add(MaxPool())
classifier.add(FastConv(64, (3, 3)))
classifier.add(MaxPool())
classifier.add(FastConv(64, (3, 3)))
classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Dense(10, activation='softmax'))
classifier.summary()
classifier.compile(crossentropy.SparseCategoricalCrossentropyLoss, adam.Adam(1e-3, beta1=0.9, beta2=0.999, eps=0.01), 'categorical_acc')

# weights = np.load("C:/Users/jurek/Desktop/Studium/Semester 5/CNN/saves/classifier_08.npz")
# classifier.set_weights(list(weights.values()))

classifier.fit(train_images, train_labels[:, 0], test_images, test_labels[:, 0],
               epochs=10, batch_size=1, verbose=True, train_tol=None, test_tol=0.90, save_update_ratio=True,
               fname="C:/Users/jurek/Desktop/Studium/Semester 5/CNN/saves/classifier_08-2")

update_ratios = classifier.history["update_ratio"]
update_ratio_1 = update_ratios[list(update_ratios.keys())[0]]
update_ratio_2 = update_ratios[list(update_ratios.keys())[1]]
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.plot(range(1, len(classifier.history['train_loss'])+1), classifier.history['train_loss'], 'r-', label="train_loss")
ax1.plot(range(1, len(classifier.history['train_acc'])+1), classifier.history['train_acc'], 'b-', label="train_acc")
ax1.plot(range(1, len(classifier.history['test_loss'])+1), classifier.history['test_loss'], 'r--', label="test_loss")
ax1.plot(range(1, len(classifier.history['test_acc'])+1), classifier.history['test_acc'], 'b--', label="test_acc")
ax2.plot(range(1, len(update_ratio_1)+1), update_ratio_1, 'yx--', label="ratio_1")
ax2.plot(range(1, len(update_ratio_2)+1), update_ratio_2, 'bx--', label="ratio_2")
ax2.set_xticks(range(1, len(update_ratio_1), (len(update_ratio_1)+1)//10))
plt.legend(loc="best")
plt.show()
