import json
import logging
import os
import shutil
import time

import numpy as np
import pyttsx3
import sounddevice as sd
from faster_whisper import WhisperModel

from google import genai
import objc
from google.genai import types
import re
import os

# from Robot_Vision.facial_recognition import FacialRecoginition
from logging_config import init_logging, resolve_debug_flag

# Path for user conversation locations
DB_PATH = os.path.join(os.getcwd(), "database_llm")

def create_fresh_unknown_id(base_dir="database_llm"):
    """
    Always create a brand-new folder unknown_<id>,
    regardless of whether the person was seen before.
    """
    os.makedirs(base_dir, exist_ok=True)

    pattern = re.compile(r"^unknown_(\d+)$")
    ids = []

    for name in os.listdir(base_dir):
        m = pattern.match(name)
        if m:
            ids.append(int(m.group(1)))

    next_id = max(ids, default=0) + 1
    folder_name = f"unknown_{next_id}"
    folder_path = os.path.join(base_dir, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    return folder_path, folder_name

class Conversation(object):

    def __init__(self, conversation_path="", debug=None):
        # self.api_key = "AIzaSyBXuSdBPzGOUBXOz71ur_2DpDksHSF3s-k"
        self.debug = init_logging(debug)
        self.stt_result = ""
        self.engine = pyttsx3.init('nsss')
        self.engine.setProperty('rate', 120)
        voices = self.engine.getProperty('voices')
        #self.engine.setProperty('voice', voices[1].id)

        folder_path, folder_name = create_fresh_unknown_id()
        conversation_path = os.path.join(folder_path, "conversation.json")

        with open(conversation_path, "w") as f:
            json.dump({"messages": []}, f, indent=2)

        self.conversation_path = conversation_path
        self.conversation_name = folder_name
        self._update_logger_name()
        self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        
        # Audio calibration state
        self.default_sample_rate = 16000
        self.calibrated_noise_floor = None
        self.calibrated_threshold = None
        self.calibrated_sample_rate = None
        self.calibration_ms = None
        
        # Starting prompts for conversations
        prompt = {"role": "user", "parts": [{"text": """
                            System Instruction: You are a gentle robot looking conduct reminiscence therapy with someone to have a meaningful conversation.
                            Start off with a questions about hometown, important people, key events, music, or memories.
                            Then continue with light hearted and friendly conversation from there, trying to understand and relate with the user. 
                            You are allowed to be empathetic and curious, but only provide appropiate responses.
                            You are not allowed to ask about politics, religion, or dating avenues. Too personal could be offputting.
                            Make sure to keep responses within 2 sentences!!
                            Please start us off with a question!
                            """}]}
        
          # Collect Conversation History
        with open(self.conversation_path, "r") as f:
            data = json.load(f)
        self.conversation = data["messages"]
        self.logger.info(
            "Conversation initialised for %s (path=%s, debug=%s).",
            self.conversation_name,
            self.conversation_path,
            self.debug,
        )

        self.conversation.append(prompt)
        self.remember_conversation() 
        self.conversation_path = conversation_path
        
        # Avoid asking about politics, religion, and dating avenues. Too personal could be offputting.
        
        # LLM Setup
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            self.logger.warning("API Key not set within environment.")
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"
        
        # Audio Calibration Setup
        try:
            self.calibrate_noise_floor(duration=1.0, sample_rate=self.default_sample_rate)
        except Exception:
            self.logger.warning("Noise floor calibration failed during init; will attempt later when recording.")

    def _update_logger_name(self):
        suffix = "Unknown"
        self._logger_name = f"{__name__}.Conversation[{suffix}]"

    @property
    def logger(self):
        init_logging(self.debug)
        name = getattr(self, "_logger_name", f"{__name__}.Conversation")
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        return logger
    
    def testing(self, client):
        model = self.model
        prompt = {"role": "user", "parts": [{"text": """
                            System Instruction: You are a gentle robot looking conduct reminiscence therapy with someone to have a meaningful conversation.
                            Start off with a questions about hometown, important people, key events, music, or memories.
                            Then continue with light hearted and friendly conversation from there, trying to understand and relate with the user. 
                            You are allowed to be empathetic and curious, but only provide appropiate responses.
                            You are not allowed to ask about politics, religion, or dating avenues. Too personal could be offputting.
                            Make sure to keep responses within 2 sentences!!
                            Please start us off with a question!
                            """}]}
        response = client.models.generate_content(model=model, contents=prompt)

        print(response.text)
    
    def llm_chat(self, skip=False):
        """
        Chat with the LLM and return the response
        """
        if skip:
            return "This is a test to see if the TTS is"
        self.logger.debug("Sending prompt to LLM")
        self.logger.debug("Sending Content: {self.conversation}")
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.conversation,
            config=types.GenerateContentConfig(
                #thinking_config=types.ThinkingConfig(thinking_budget=-1, include_thoughts=True)
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        content = response.text
        self.conversation.append({'role': 'model', 'parts': [{"text": content}]})
        self.logger.debug("LLM response: %s", content)
        return content

    def record_and_transcribe(self, duration=2, sample_rate=16000):
        """
        Record audio and transcribe it to text
        """
        # self.logger.info("Recording audio for %.2fs at %sHz.", duration, sample_rate)
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        self.logger.debug("Transcribing recorded audio.")
        segments, _ = self.whisper_model.transcribe(audio.flatten(), beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        self.logger.debug("Transcription result: %s", text)
        return text

    def calibrate_noise_floor(self, duration=1.0, sample_rate=16000):
        """
        Calibrate the noise floor
        """
        self.logger.info("Calibrating noise floor for %ss at %sHz. Please remain silent.", duration, sample_rate)
        chunk_ms = 50
        chunk_size = int(sample_rate * (chunk_ms / 1000.0))
        frames_rms = []
        start_time = time.time()
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
            while (time.time() - start_time) < duration:
                data, _ = stream.read(chunk_size)
                if data is None:
                    continue
                mono = data.reshape(-1)
                rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
                frames_rms.append(rms)
        if len(frames_rms) == 0:
            raise RuntimeError("No audio captured during calibration")
        noise_floor = float(np.median(frames_rms))
        threshold = max(noise_floor * 2.5, 1e-3)
        self.calibrated_noise_floor = noise_floor
        self.calibrated_threshold = threshold
        self.calibrated_sample_rate = sample_rate
        self.calibration_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "Noise floor calibrated in %sms: noise_floor=%.6f, threshold=%.6f",
            self.calibration_ms,
            noise_floor,
            threshold,
        )

    def record_and_transcribe_until_silence(
        self,
        min_duration=5.0,
        silence_duration=5.0,
        max_duration=30.0,
        sample_rate=16000,
    ):
        """
        Record audio and transcribe it to text until silence is detected
        """
        # Ensure calibration for this sample rate
        if self.calibrated_threshold is None or self.calibrated_sample_rate != sample_rate:
            self.logger.debug("No calibration for sample_rate=%s. Calibrating now...", sample_rate)
            try:
                self.calibrate_noise_floor(duration=0.5, sample_rate=sample_rate)
            except Exception:
                self.logger.warning("Calibration failed; proceeding with conservative default threshold.")
                self.calibrated_noise_floor = 1e-3
                self.calibrated_threshold = 1e-3
                self.calibrated_sample_rate = sample_rate

        self.logger.info(
            "Recording until silence (min_duration=%.2f, silence_duration=%.2f, max_duration=%.2f, sample_rate=%s).",
            min_duration,
            silence_duration,
            max_duration,
            sample_rate,
        )
        chunk_ms = 50
        chunk_size = int(sample_rate * (chunk_ms / 1000.0))
        frames = []
        start_time = time.time()
        last_voice_time = start_time

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
            while True:
                data, _ = stream.read(chunk_size)
                if data is None:
                    continue
                mono = data.reshape(-1)
                frames.append(mono.copy())

                rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))

                elapsed = time.time() - start_time
                threshold = self.calibrated_threshold if self.calibrated_threshold is not None else 1e-3

                if rms > threshold:
                    last_voice_time = time.time()

                # Stop if recorded long enough and observed sustained silence
                if elapsed >= min_duration and (time.time() - last_voice_time) >= silence_duration:
                    break

                # Hard stop at max_duration
                if elapsed >= max_duration:
                    break

        # Concatenate and measure actual recorded duration
        if len(frames) == 0:
            self.logger.warning("No audio captured during silence-detection recording.")
            return ""

        audio = np.concatenate(frames)
        recorded_seconds = len(audio) / float(sample_rate)
        self.logger.info("Recorded duration: %.2fs", recorded_seconds)

        # Transcribe
        self.logger.debug("Transcribing captured audio.")
        t0 = time.time()
        segments, _ = self.whisper_model.transcribe(audio, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        transcribe_seconds = time.time() - t0
        self.logger.debug("Transcription completed in %.2fs. Result: %s", transcribe_seconds, text)
        return text

    def text_to_speech(self, response):
        """
        Text to speech the response
        """
        # import pyttsx3
        # self.logger.debug("Speaking response via TTS.")

        # self.engine = pyttsx3.init()
        self.engine.say(response)
        self.engine.runAndWait()

    
    def remember_conversation(self):
        """
        End the conversation and save the conversation history
        """
        with open(self.conversation_path, "w") as f:
            json.dump({"messages": self.conversation}, f, indent=2)
        self.logger.info("Conversation history written to %s.", self.conversation_path)

            
    def main(self):
        print("Starting conversation")
        init_logging(self.debug)
        response = self.llm_chat()
        self.logger.info("[LLM]: %s", response)
        self.text_to_speech(response)
        
        print("recording")
        self.logger.info("Conversation loop starting for %s (debug=%s).", self.conversation_name, self.debug)
        while True:
            self.stt_result = self.record_and_transcribe_until_silence()
            if self.stt_result == "" or self.stt_result == " ":
                self.logger.warning("No response detected from user during conversation.")
                self.engine.say("Sorry I couldn't catch that, could you please say that again?")
                self.engine.runAndWait()
                continue
            if "done" in self.stt_result.lower() or "quit" in self.stt_result.lower() or "end conv" in self.stt_result.lower() or "and conv" in self.stt_result.lower():
                self.logger.info("User requested to end the conversation.")
                break
            self.conversation.append({'role': 'user', 'parts': [{"text": self.stt_result}]})
            self.logger.info("[User]: %s", self.stt_result)
            
            responding = self.llm_chat()
            self.logger.info("[LLM]: %s", responding)
            self.text_to_speech(responding)
            self.logger.info("[LLM]: Responded TTS")

            self.remember_conversation()

        
        self.logger.info("Conversation loop concluded for %s.", self.conversation_name)
    
if __name__ == "__main__":
    conversation_path = os.path.join(DB_PATH)
    conversation = Conversation(conversation_path=conversation_path)
    conversation.main()