## Urban Sound Classification 

<p><a href="https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20NN.ipynb">
Python notebook </a> for blog post <a href="http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/">Urban sound classification using Neural Network</a>.</p>

<p><a href="https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20CNN.ipynb">Python notebook</a> for blog post <a href="http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/">Urban sound classification using Convolutional Neural Network</a>.</p>

<p><a href="https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20RNN.ipynb">Urban sound classification using RNN</a>.</p>

### Project Goal:

This project uses laptop microphones and a vibrating Myo wristband to alert its wearer when a siren is detected.
This is useful for deaf or hearing impaired drivers and pedestrians.
It uses a small neural network based on the above blog posts to classify sound recorded from a microphone, sending a bluetooth signal to the myo wristband when
a siren is detected. It currently works with 91% accuracy.

### Tools Required

* Python 3.5
* Tensorflow
* Numpy
* Librosa
* myo sdk
* <a href="https://github.com/NiklasRosenstein/myo-python">myo-python</a> bindings for myo sdk
* pyaudio

### Dataset

The UrbanSound8k dataset used for model training, can be downloaded from the following [[link]](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html).

Another related dataset that can be used is Google's [AudioSet](https://research.google.com/audioset/).

### Future steps:

* Standalone Android app
* Use a wristband with a microphone instead of myo
* Improve the neural network performance
* Improve the audio preprocessing
* Get more data, and try improving data (especially for training with vocal interference)