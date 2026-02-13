import os
import threading
import random
import time

try:
    from playsound import playsound
except ImportError:
    print("The playsound module is not installed. Please install it with 'pip install playsound'.")

class AudioManager:
    def __init__(self, starting_voice_type, audio_folder_path, audio_file_encoding=".wav"):
        self._voice_type = starting_voice_type
        self._audio_folder_path = audio_folder_path
        self._audio_file_encoding = audio_file_encoding
        self.build_audio_path_end()
        self.rng = random.Random()
        self._last_audio_clip = None

    def build_audio_path_end(self):
        # Builds the suffix for audio file names, e.g. "_DefaultVoice.wav"
        self._audio_path_end = "_" + self._voice_type + self._audio_file_encoding

    @property
    def audio_file_type(self):
        return self._audio_file_encoding

    @audio_file_type.setter
    def audio_file_type(self, value):
        self._audio_file_encoding = value
        self.build_audio_path_end()

    @property
    def selected_voice(self):
        return self._voice_type

    @selected_voice.setter
    def selected_voice(self, value):
        self._voice_type = value
        self.build_audio_path_end()

    def send_audio(self, clip_name, async_=True, isvoice=True, encoding=""):
        """
        Constructs the full path to the audio clip and plays it.
        If async_ is True, the audio is played in a separate thread.
        """
        if isvoice:
            clip_name = clip_name+"_"+self._voice_type
        if encoding != "":
            clip_name = clip_name + encoding
        else:
            clip_name = clip_name + self._audio_file_encoding
        full_path = os.path.join(self._audio_folder_path, clip_name)
        self._last_audio_clip = full_path
        print(f"Playing audio: {full_path}")

        if async_:
            thread = threading.Thread(target=self._play_audio, args=(full_path,))
            thread.start()
        else:
            self._play_audio(full_path)

    def _play_audio(self, full_path):
        try:
            playsound(full_path)
        except Exception as e:
            print(f"Error playing audio '{full_path}': {e}")

    def repeat_audio(self):
        if self._last_audio_clip:
            print(f"Repeating audio: {self._last_audio_clip}")
            self._play_audio(self._last_audio_clip)
        else:
            print("No audio to repeat.")


    def send_audio_with_probability(self, clip_name, play_probability=1.0, async_=True, isvoice=True, encoding=""):
        """
        Plays the audio clip based on the given play probability.
        Returns True if the audio was played, or False if skipped.
        """
        if self.rng.random() < play_probability:
            self.send_audio(clip_name, async_, isvoice, encoding)
            return True
        else:
            print(f"Skipping audio '{clip_name}' due to probability check.")
            return False

    def process_audio_call(self, audio_clip):
        """
        Parses and appropriately plays an audio file based on instructions from a keyframe
        Takes in the AudioClip section of the keyframe, as a set of key,value pairs
        Returns True if the audio was played, or False if skipped.
        """
        # Get our clip values, and check that we actually have a clip name at all
        clip_name = audio_clip.get("ClipName","")
        if clip_name == "": return False
        async_ = audio_clip.get("Async","False") == "True"
        probability = audio_clip.get("Probability",1)
        isvoice = audio_clip.get("IsVoice","True") == "True"
        encoding = audio_clip.get("Encoding","")

        if audio_clip.get("IsGroup","False") == "True":
            # then handle it like a group of audio options rather than a standalone clip
            return False
        else:
            # If the clip isn't part of a group, then send it as a standalone audio clip with a probability
            return self.send_audio_with_probability(clip_name, probability, async_, isvoice, encoding)

        

# if __name__ == "__main__":
#     # Example usage:
#     # Replace the following folder with the path where your audio files are stored.
#     audio_folder = "path/to/your/audio_files"
#     # Create an AudioManager with a starting voice (e.g., "DefaultVoice")
#     audio_manager = AudioManager(starting_voice_type="DefaultVoice", audio_folder_path=audio_folder)
    
#     # Play some sample audio clips.
#     audio_manager.say_hello()
#     time.sleep(2)
#     audio_manager.say_yes()
#     time.sleep(2)
#     audio_manager.say_no()
#     time.sleep(2)
#     audio_manager.say_laugh()
#     time.sleep(2)
#     audio_manager.say_goodbye()
