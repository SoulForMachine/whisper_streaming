import wave
import numpy as np

class WavWriter:
    def __init__(self, filename, sample_rate=16000):
        """
        Open a WAV file for streaming writes.
        """
        self.sample_rate = sample_rate
        self.wf = wave.open(filename, "wb")
        self.wf.setnchannels(1)           # mono
        self.wf.setsampwidth(2)           # 16-bit PCM
        self.wf.setframerate(sample_rate)

    def write_chunk(self, chunk: np.ndarray):
        """
        Write a single audio chunk (float32 in [-1,1] or int16).
        """
        if chunk.dtype != np.int16:
            # Convert float32 → int16
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * 32767).astype(np.int16)

        self.wf.writeframes(chunk.tobytes())

    def close(self):
        """
        Finalize and close the WAV file.
        """
        self.wf.close()
