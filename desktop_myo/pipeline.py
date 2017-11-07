from time import sleep
import tensorflow as tf
import pyaudio
import wave
from predictNew import classify

"""
Pipeline:

 -record audio from mic with pyaudio
 -save as short .wav clips
 -(apply fft)
 -give to trained neural network
 -if classified as siren, run myo script

Note:
 -in conda prompt, activate with "activate tensorflowc"

"""
# ----------------------- Audio setup --------------------
#initializes audio and provides functions for recording from the stream and closing the stream
def audio_init():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	WAVE_OUTPUT_FILENAME = 'output.wav'

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
				channels=CHANNELS,
				rate=RATE,
				input=True,
				frames_per_buffer=CHUNK)

	def get_audio(seconds, file):
		print('recording')
		frames = []
		for i in range(0, int(RATE / CHUNK * seconds)):
			data = stream.read(CHUNK)
			frames.append(data)
		print('done recording')
		wf = wave.open(file, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(p.get_sample_size(FORMAT))
		wf.setframerate(RATE)
		wf.writeframes(b''.join(frames))
		wf.close()
		return file

	def close_audio():
		stream.stop_stream()
		stream.close()
		p.terminate()

	return (get_audio, close_audio)

    
# ------------- Begin main code ---------------------
audio_get, audio_close = audio_init()

i = 0
while (i < 50):
	uniquename = 'output' + str(i % 3) + '.wav'
	print(uniquename)
	file = audio_get(6, uniquename)
	print('file is ', file)
	if classify(file):
		myo_vibrate()
		print("-----------------found it!----------------")
	i += 1

audio_close()