from time import sleep
import tensorflow as tf
import pyaudio
import wave
from myo import init, Hub, DeviceListener
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
	#RECORD_SECONDS = 5
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


# ------------- For Myo interaction -------------------
class Listener(DeviceListener):

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")
        myo.vibrate('long')

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")
		
def myo_init():
	init()
	hub = Hub()
	def vibrate():
		hub.run(500, Listener())
		hub.shutdown()
	def close():
		hub.shutdown()
	return (vibrate, close)
	
	
# ------------- Preprocessing -----------------------
	
# ------------- Begin main code ---------------------

audio_get, audio_close = audio_init()

myo_vibrate, myo_close = myo_init()


"""
if predictNew.classify("../audio/fold1/106905-8-0-2.wav"):
	print ('yes')
	myo_vibrate()
	
print('huh')

"""



i = 0
while (i < 50):
	#TODO: implement multithreading, buffering
	uniquename = 'output' + str(i % 10) + '.wav'
	print(uniquename)
	file = audio_get(5, uniquename)
	print('file is ', file)
	if classify(file):
		myo_vibrate()
		print("-----------------found it!----------------")
	i += 1

audio_close()

myo_close()
