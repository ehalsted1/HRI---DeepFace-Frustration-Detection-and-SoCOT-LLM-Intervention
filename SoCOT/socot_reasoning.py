import json
import logging
import os
import shutil
import time

import numpy as np

from google import genai
import objc
from google.genai import types
import re
import os
import threading 
from enum import Enum, auto

from queue import Queue
from logging_config import init_logging, resolve_debug_flag

class Reasoning:
    '''
    this class takes in the primitives class, and json object of the combined primitives, and 
    applies reasoning to them to extract the desired check in type 
    the model employs a chain of thought reaosning prompt 
    Based on the paper: " 
    
    
    '''
    def create_fresh_unknown_id(self, base_dir="database_socot"):
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
    
    
    def __init__(self, debug=None, result_path=""):
        #initialize model
        self.debug = init_logging(debug)

        #define prompt
        prompt = {'role': 'user', 'parts': [{"text": """Using step by step reasoning: Based on the information given to you (voice pitch, emotions detected, and speech), 
                                             what detected affect state of the person: {neutral, positive, negative}. 
                                              Then based on this answer, provide a one sentence check in question.
                                              Please answer in the format: {Check in Type}: {Check in question}"""}]}

        #create folder and json for prompting

        folder_path, folder_name = self.create_fresh_unknown_id()
        result_path = os.path.join(folder_path, "result.json")

        with open(result_path, "w") as f:
            json.dump({"results": []}, f, indent=2)

        self.result_path = result_path
        self.result_name = folder_name
        self._update_logger_name()

        # collect prompt history 
        with open(self.result_path, "r") as f:
            data = json.load(f)
        self.result = data["results"]

        self.logger.info(
            "Process initialised for %s (path=%s, debug=%s).",
            self.result_name,
            self.result_path,
            self.debug,
        )

        self.result.append(prompt)
        self.remember_conversation()
        self.result_path = result_path

        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            self.logger.warning("API Key not set within environment.")
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash-lite"

    def _update_logger_name(self):
        suffix = "Unknown"
        self._logger_name = f"{__name__}.Results[{suffix}]"

    @property
    def logger(self):
        init_logging(self.debug)
        name = getattr(self, "_logger_name", f"{__name__}.Conversation")
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        return logger

    def parse_json_file(self, json_path):
        '''
        passing in primitive file from output/combined primitive'''

        pass

    def remember_conversation(self):
        """
        End the conversation and save the conversation history
        """
        with open(self.result_path, "w") as f:
            json.dump({"results": self.result}, f, indent=2)
        self.logger.info("Conversation history written to %s.", self.result_path)

    def apply_reasoning(self, json_path, skip=True):
        with open(json_path, "r") as f:
            primitive_data = json.load(f)
        
        primitive_str = json.dumps(primitive_data, indent=2, ensure_ascii=False)

        self.logger.debug("Adding in primtive data to content")
        
        # self.result.append({'role': 'user', 'parts': [{"extracted primtives": primtive_str}]})
        self.result.append({
        "role": "user",
        "parts": [{
            "text": (
                "Here are extracted primitives in JSON.\n"
                "Use them in your reasoning and reference specific fields when relevant.\n\n"
                "```json\n"
                f"{primitive_str}\n"
                "```"
                )
            }]
        })
        if not isinstance(self.result, list):
            raise TypeError(f"self.result must be a list, got {type(self.result)}")
        if skip:
            return "This is a test"
        
        self.logger.debug("Sending prompt to LLM")
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.result,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        content = response.text
        self.result.append({'role': 'model', 'parts': [{"text": content}]})
        self.logger.debug("LLM Response: %s", content)

        return content
    
    def main(self):
        content = self.apply_reasoning("output/combined_primitive_50_50_35.json", skip=False)
        print(content)
        self.remember_conversation()

if __name__ == "__main__":
    DB_path = os.path.join(os.getcwd(), "socot_output")
    reasoning = Reasoning(result_path=DB_path)
    reasoning.main()

