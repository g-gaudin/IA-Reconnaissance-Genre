import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# dataset import
dataset = pd.read_csv('data/voice.csv')

# Preprocessing of data : We normalize X and transform y classes into a binary array
X = dataset.iloc[:, :20].values
y = dataset.iloc[:, 20:21].values
sc = StandardScaler()
X = sc.fit_transform(X)
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# We split data into a training set and a test set with 10% ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Creation of the neural network
model = Sequential([
    Dense(10, input_dim=20), Activation('relu'),
    Dense(2), Activation('sigmoid'), ])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training neural network with 100 iterations
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=64)

# Visualisation of the gain in accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Visualisation of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

'''
# Test of the final result
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
'''