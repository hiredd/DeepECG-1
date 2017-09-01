DeepECG
---
Using convolutional neural networks on ECG data to see if it is a valid candidate for biometric authentication. 

### Pre-Requisites

Make sure you have the following items installed:
1. python2.7
2. [wfdb package](https://pypi.python.org/pypi/wfdb) using `pip install wfdb`
3. [keras](https://keras.io/#installation/)
4. [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
5. [numpy](https://docs.scipy.org/doc/numpy/user/install.html)

### Usage

1. `python data_processing.py`: Converts all .dat files in `data/` to .csv. Extract labels and features from individual .csv
files. Outputs the following files in the `processed_data/` folder:
    * [`ecgdblabels.csv`](https://goo.gl/4yZ19G): labels [person, age, gender, date record was collected]
    * [`filecgdata.csv`](https://goo.gl/LL4T8q): filtered ecg signals
    * [`unfilecgdata.csv`](https://goo.gl/SW85aH): unfiltered ecg signals [noisy]
    * [`new_data.csv`](https://goo.gl/wjP19o): signals aligned by first positive maximum peak [so that all signals start at the same point]
    * [`rsampled_data.csv`](https://goo.gl/xjiouP): signals aligned peak to peak and resampled to maintain width [generating more data to solve the problem of overfitting]    

##### [Note]: If the folder, `processed_data/`, does not exist in your working directory on your local machine, create it. You can either run data_processing.py to generate the .csv files mentioned above or download them from the links provided. 

2. `python model_personid.py`: Run cnn model on training data. Evaluate model. 

### Network

Input shape: (6073, 9077)

Layer 1: filters=16, shape=(1,5), activation='tanh'

Layer 2: 2DMaxPooling(size=(1,3))

Layer 3: filters=32, shape=(1,5), activation='tanh'

Layer 4: 2DMaxPooling(size=(1,3))

Flatten Layer 4.

Layer 5: size=128, activation='tanh'

Layer 6: size=90, activation='softmax'

This is the layer that spits out probabilities for each of the 90 categories (persons).

I used RMS Prop as an optimiser and Early Stopping as a regulariser with patience 10. 

### Performance
![graph](/images/performance_summary_graph.png?raw=true)

Validation Accuracy: 91%

Validation Loss: 0.546

### Database

This project uses the [ECG-ID Database from
Physionet](https://physionet.org/physiobank/database/ecgiddb). It can be found
in the `data/` folder. [Note: the original database does not include .csv files.
See `class csvGenerator` in `data_processing.py` to learn more about how the .dat files were converted to
csv using [rdsamp](https://pypi.python.org/pypi/wfdb)]
