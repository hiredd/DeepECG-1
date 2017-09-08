r""" 
Author: Aishni Parab
File: model_genderid.py
Description: calls functions in data_processing to init data. runs training and testing on data.
"""
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers import Dense

import matplotlib.pyplot as plt
import data_processing as data 
import pandas as pd
import numpy as np
import os

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data
dataset = data.getData() #instance 
X, Y, p = dataset.get()

X = X.reshape(6733, 430)

# baseline model
def create_baseline():
  # create model
  model = Sequential()
  model.add(Dense(430, input_dim=430, kernel_initializer='normal', activation='relu'))
  model.add(Dense(215, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

model= create_baseline()

#model = model.load_weights(os.path.join('saved_models', 'gen_kcv_.h5'))

# evaluate baseline model with standardized dataset
#np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

predicted = cross_val_predict(pipeline, X, Y, cv=kfold)
df_predicted = pd.DataFrame(predicted)
df_predicted.to_csv(os.path.join('saved_models', 'predicted.csv'), index=False)
print(accuracy_score(Y, predicted))

results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# save model to dir
model.save_weights(os.path.join('saved_models', 'gen_kcv_.h5'))

# load predictions from file
pred = pd.read_csv(os.path.join('saved_models', 'predicted.csv'))
pred = pd.DataFrame(pred)
predicted = np.asarray(pred)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y, predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_fn = data.plotHelper()
plt.figure()
plot_fn.plot_confusion_matrix(cnf_matrix, classes=['Male', 'Female'],
                      title='Confusion matrix')