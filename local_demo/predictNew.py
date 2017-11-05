import tensorflow as tf
import glob
import numpy as np
import librosa
import os
import sys

def extract_feature(file_name):
	#print("\nExtracting feature from " + file_name + "...")
	X, sample_rate = librosa.load(file_name)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)    #no this
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)    #no this
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)         #this might be okay
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)     #



	return mel,contrast


def parse_audio_files(file_name, file_ext='.wav'):
	features = np.empty((0,135))
	fn = file_name
	mel, contrast = extract_feature(fn)
	ext_features = np.hstack([mel,contrast])
	features = np.vstack([features,ext_features])
	return np.array(features)

def classify(testfilepath):
	features_test = parse_audio_files(testfilepath)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph('./nn_model.meta')
		saver.restore(sess, "./nn_model")

		graph = tf.get_default_graph()
		X = graph.get_tensor_by_name("X:0")
		Y = graph.get_tensor_by_name("Y:0")
		h_1 = graph.get_tensor_by_name("h_1:0")
		h_2 = graph.get_tensor_by_name("h_2:0")
		y_ = graph.get_tensor_by_name("y_:0")


		y_pred = sess.run(tf.argmax(y_,1),feed_dict={X : features_test})
		print(y_pred[0])
		if int(y_pred[0]) is 8:
			return True
		else:
			return False
