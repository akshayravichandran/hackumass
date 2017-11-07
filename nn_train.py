import glob
import os
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
#%matplotlib inline
plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

###############################################################################

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
"""
def load_sound_files(file_paths):
    print("Load Sound File Entered....")
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    print("\nFiles successfully loaded!")
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

###############################################################################

sound_file_paths = ["57320-0-0-7.wav","24074-1-0-3.wav","15564-2-0-1.wav","31323-3-0-1.wav","46669-4-0-35.wav",
                   "89948-5-0-0.wav","40722-8-0-4.wav","103074-7-3-2.wav","106905-8-0-0.wav","108041-9-0-4.wav"]
sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
               "gun shot","jackhammer","siren","street music"]

raw_sounds = load_sound_files(sound_file_paths)
"""
###############################################################################

#plot_waves(sound_names,raw_sounds)
#plot_specgram(sound_names,raw_sounds)
#plot_log_power_specgram(sound_names,raw_sounds)

###############################################################################

def extract_feature(file_name):
    #print("\nExtracting feature from " + file_name + "...")
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))             #short time fourier transform -- useful!
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)   #features based on chromatic scale -- for music analysis
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mel,contrast

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,135)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mel, contrast = extract_feature(fn)
            ext_features = np.hstack([mel,contrast])
            features = np.vstack([features,ext_features])
            lbl = fn.split('/')[2].split('-')[1]
            labels = np.append(labels, 1 if lbl is '8' else 0)
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

###############################################################################

#features_test, labels_test = parse_audio_files('audio',['foldtest'])
#print(labels_test)

parent_dir = 'audio_balanced'

sub_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
features, labels = parse_audio_files(parent_dir,sub_dirs)

###############################################################################

labels = one_hot_encode(labels)

train_test_split = np.random.rand(len(features)) < 0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]

###############################################################################

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

###############################################################################

training_epochs = 5000
n_dim = features.shape[1]
n_classes = 2
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

###############################################################################

X = tf.placeholder(tf.float32,[None,n_dim], name='X')
Y = tf.placeholder(tf.float32,[None,n_classes] ,name='Y')

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd), name='W_1')
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd), name='b_1')
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1, name="h_1")


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd), name='W_2')
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd), name='b_2')
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2, name='h_2')


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name='W')
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name='b')
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b, name='y_')

init = tf.global_variables_initializer()

###############################################################################

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################################################

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        print("Epoch", epoch, "\n")
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,cost)
    #y_pred = sess.run(tf.argmax(y_,1),feed_dict={X : features_test})
    #print("y_pred: ", y_pred)
    saver = tf.train.Saver()
    saver.save(sess,"./local_demo/nn_model")
    #y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
    #y_true = sess.run(tf.argmax(test_y,1))


###############################################################################

#fig = plt.figure(figsize=(10,8))
#plt.plot(cost_history)
#plt.ylabel("Cost")
#plt.xlabel("Iterations")
#plt.axis([0,training_epochs,0,np.max(cost_history)])
#plt.show()

#p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
#print ("F-Score:", round(f,3))
