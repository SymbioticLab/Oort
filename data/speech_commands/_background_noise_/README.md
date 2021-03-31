# Background Noise Audio Files

These audio files are part of the Speech Commands data set, and covered by the same Creative Commons BY 4.0 license (see the root directory for more details).

They are a collection of sounds that can be mixed in to simulate background noise when training machine learning models on audio clips. They were all either collected personally by Pete Warden in July 2017, or generated in the case of the pink and white noise. Those noise samples were created using the following lines of Python code:

scipy.io.wavfile.write('/tmp/white_noise.wav', 16000, np.array(((acoustics.generator.noise(16000*60, color='white'))/3) * 32767).astype(np.int16))
scipy.io.wavfile.write('/tmp/pink_noise.wav', 16000, np.array(((acoustics.generator.noise(16000*60, color='pink'))/3) * 32767).astype(np.int16))
