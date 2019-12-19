import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import subprocess
import sys
import os

if os.path.exists("wavs/results.csv"):
    os.remove("wavs/results.csv")
if os.path.exists("wavs/test-p1.tiff"):
    os.remove("wavs/test-p1.tiff")
arg = ["wavs"]
command = 'Rscript'

cmd = [command,'scriptsR/voiceAnalyzer.R'] + arg
subprocess.check_call(cmd, shell=False) #Run WarbleR script

#Now checking all extracted parameters
table = []
with open('wavs/results.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'{", ".join(row[3:9])}'+','+f'{", ".join(row[13:16])}'+','+f'{", ".join(row[18:24])}')
            line_count += 1
        else:
            col_count = 0
            line = []
            for col in row:
                if (col_count > 2 and col_count != 9 and col_count!=10 and col_count!=11 and col_count!=12 and col_count!=16 and col_count!=17 and col_count<24):
                    line.append(float(col))
                col_count += 1
            table.append(line)
            line_count += 1
results_np_array = np.array(table)

# dataset import
dataset = pd.read_csv('data/voice.csv')

# Preprocessing of data : We normalize X and transform y classes into a binary array
X = dataset.iloc[:, :15].values
y = dataset.iloc[:, 15:16].values
sc = StandardScaler()
X = sc.fit_transform(X)
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print(y[0])

# We split data into a training set and a test set with 10% ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Creation of the neural network
model = Sequential([
    Dense(20, input_dim=15), Activation('relu'),
    Dense(20, input_dim=20), Activation('relu'),
    Dense(2), Activation('sigmoid'), ])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training neural network with 20 iterations
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=64)

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

#Prediction on test.wav results
predictions = model.predict(results_np_array)
line = 0
for row in predictions:
    sys.stdout.write(np.array2string(predictions[line][0]*100))
    sys.stdout.write("% femelle - ")
    sys.stdout.write(np.array2string(predictions[line][1]*100))
    print("% male")
    line+=1