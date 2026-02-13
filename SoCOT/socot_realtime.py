import cv2
import logging
import os
import re
import time

import numpy as np
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from faster_whisper import WhisperModel

import json
import parselmouth
import librosa
from pathlib import Path

import logging
import pandas as pd 

from moviepy.editor import VideoFileClip 

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PrimitiveExtractionRT:
    def __init__(self) -> None:
        self.reasoning_model = None
        self.camera = 0
        self.id = None
    
    def get_file_id(self, conversation_path):
        '''returns participant ID from format ~~/unknown_{id}'''
        pattern = r".*/unknown_(?P<id>[^/]+)$"
        
        match = re.search(pattern, conversation_path)
        if match:
            id = match.group("id")
            return id
        else:
            return None


    def recording_b4_after(self, recording=False):
        '''
        this function records at the very begginning of the listening , and resumes recording for 1 second after after terminated
        and return the mp4 file 
        '''
        while recording:
            pass


    def real_time_passing(self, recording=False):
        pass
    
    def real_time_audio():
        pass

    

    def main(self):
        conversation_path = '/Users/loni/Documents/OSU/SHARE_Lab/Reminiscence_multithreading_integration/database_llm/unknown_1'
        print(self.get_file_id(conversation_path))

if __name__ == "__main__":
    test = PrimitiveExtractionRT()
    test.main()
