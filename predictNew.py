import tensorflow as tf
import glob
import numpy as np
import librosa
import os




def extract_feature(file_name):
    #print("\nExtracting feature from " + file_name + "...")
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    print("\nParse audio files entered...")
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('\\')[2].split('-')[1])
    print("\nAudio files successfully parsed")
    return np.array(features), np.array(labels, dtype = np.int)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./nn_model.meta')
    saver.restore(sess, "./nn_model")


    features_test, labels_test = parse_audio_files('audio','foldtest')

    print(tf.trainable_variables())

    y_pred = sess.run([tf.argmax(y_,1)],feed_dict={X: features_test})
    y_true = sess.run(tf.argmax(labels_test,1))
    print("y_pred: ", y_pred)
    print("y_true: ",y_true)
