import speech_recognition as sr
from ollama import generate
import os
import time
from gtts import gTTS
import pygame
from tempfile import NamedTemporaryFile
import serial
import json
import threading
from threading import Event
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import queue
from queue import Empty
import re
import requests
import random

language = "en"

#define Ollam model and start from empty convo history
os.environ["OLLAMA_MODELS"] = os.path.abspath("ollama/models")
conversation_history = ""

#defining arduino port + starting sami files (assuming softwafre folder is accesible as is )
#TODO: update arduino ports 
arduino_port='/dev/tty.usbserial-1130'
baud_rate=115200
joint_config_file='config/Joint_config.json'
emote_file='config/Emote.json'
audio_folder='audio'
starting_voice='Matt'
audio_file_encoding='.mp3'

#defining emote paths
dir_path = "emotes"
available_emotes = []
for json_file in [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]:
    available_emotes.append("emotes/"+json_file)
    print("emotes/"+json_file)

#openikng starting files
with open(emote_file, 'r') as file:
    data = json.load(file)
emote_mapping = data["Emotes"]

with open(joint_config_file, 'r') as file:
    config = json.load(file)
joint_map = {}
for joint in config["JointConfig"]:
    joint_map[joint["JointName"]] = joint["JointID"]

full_joint_config = None
with open(joint_config_file, 'r') as f:
    full_joint_config = json.load(f)['JointConfig']
full_joint_map = {joint['JointName']: joint for joint in full_joint_config}

# starting queues (l = ) (r = )
qL = queue.Queue()
qR = queue.Queue()

print(available_emotes)
print(emote_mapping)
print(joint_map)
print(full_joint_config)
print(full_joint_map)

mic_names = sr.Microphone.list_microphone_names()
print("Available Microphones:")
for i, name in enumerate(mic_names):
    print(f"{i}: {name}")
mic_index = int(input("Enter the index of the microphone you want to use: "))

#starting server for chat (?), local multithreading server
url = "http://localhost:11434/api/chat"
SAVE_FILE = 'game_so_far.txt'
TEXT_ONLY_MODE = False
story_summary = ""
full_story = ""
recent_turns = []
inventory = {}
ser = None

########################################################################
# connect to robot controller
########################################################################
def initialize_serial_connection():
    try:
        ser = serial.Serial(arduino_port, baud_rate, timeout=1)
        time.sleep(2)
        print("Serial connection established.")
        packet = [0x3C, 0x50, 0x01, 0x45, 0x3E]
        ser.write(bytearray(packet))
        print("Sent packet:", bytearray(packet))
        if ser.in_waiting > 0:
            msg = ser.readline().decode()
            print("Arduino response:", msg)
        return ser
    except serial.SerialException as e:
        print("Error connecting to Arduino:", e)

def close_connection(ser):
    if ser:
        ser.close()
        print("Serial connection closed.")

def send_joint_command(ser, joint_ids, joint_angles, joint_time):
    if len(joint_ids) != len(joint_angles):
        raise ValueError("Mismatch in joint IDs and angles.")
    packet = [0x3C, 0x4A, joint_time]
    for jid, angle in zip(joint_ids, joint_angles):
        packet.extend([jid, angle])
    packet.append(0x3E)
    ser.write(bytearray(packet))
    #print("Sent joint command: ", bytearray(packet))
    qR.put(["Joint", str(bytearray(packet))])

def send_emote(ser, emote_id):
    packet = [0x3C, 0x45, emote_id, 0x3E]
    ser.write(bytearray(packet))
    #print("Sent emote command: ", bytearray(packet))
    qR.put(["Emote", str(bytearray(packet))])
    time.sleep(1)

def get_joint_id(joint_name):
    return joint_map.get(joint_name, 0)
    
#unique function - running json 
'''
running json from filename, the stop event is related perhaps to the conversation
per key frame in the json file 
sending emote first then joint angles
'''
def read_json(ser, filename, stop_event):
    print("Running gesture: " + str(filename))
    qR.put(["Action", "Running gesture: " + str(filename)])
    with open(filename, 'r') as file:
        data = json.load(file)
    for keyframe in data["Keyframes"]:
        if stop_event.is_set():
            print("Stopping gesture early.")
            break
        if keyframe.get("HasEmote") == "True":
            expression = keyframe.get("Expression", "Neutral")
            emote_value = emote_mapping.get(expression, 0)
            send_emote(ser, emote_value)
        # Process Joint Commands if enabled.
        if keyframe.get("HasJoints") == "True":
            joint_ids = []
            joint_angles = []
            joint_time = keyframe.get("JointMoveTime", 1)
            for joint in keyframe["JointAngles"]:
                joint_ids.append(get_joint_id(joint["Joint"]))
                joint_angles.append(joint["Angle"])
            send_joint_command(ser, joint_ids, joint_angles, joint_time)
        time.sleep(keyframe.get("WaitTime", 1000) / 1000)

# potentially used for selecting based on response
'''
looks like unused function for selecting a response based on emotion classification model

could use to classify text based response 
'''
def get_gesture(response):
    #model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    #emotion = model(response)    
    #print(emotion)
    return available_emotes[random.randint(0, len(available_emotes) - 1)]
    #return available_emotes[1]

def speech_to_text(mic_index):
    r = sr.Recognizer()
    with sr.Microphone(device_index=mic_index) as source:
        r.adjust_for_ambient_noise(source)
        failures = 0
        while True:
            print("Listening...")
            audio = r.listen(source)
            try:
                return r.recognize_google(audio)
            except sr.UnknownValueError:
                failures = failures + 1
                print("Could not understand audio. Current failures: " + str(failures), end='\033[F')
            except sr.RequestError as e:
                print(f"API error: {e}")

'''
def transmit_prompt(prompt):
    global conversation_history
    print("Sending prompt to model...", end='\r')
    qR.put(["Action", "Sending prompt to model..."])

    conversation_history += f"User: {prompt}\nNarrator:"

    request = generate(model='llama_mud', prompt=conversation_history)
    response = request['response'].strip()

    conversation_history += f" {response}\n"

    return response
'''

# sending data to llama model for reponse 
''' 
i think i will have to change this to format differently (check function format_inventory_prompt)
'''
def transmit_prompt(prompt):
    print("Sending prompt to model...", end='\r')
    qR.put(["Action", "Sending prompt to model..."])
    inventory_context = format_inventory_for_prompt()
    content = f"{inventory_context}\n{prompt}"
    
    data = {
        "model": "llama_mud",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print("Error during LLAMA call:", e)
        return "[Error]" 

def text_to_speech(response):
    ttsobj = gTTS(text=response, lang=language, slow=False)
    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        ttsobj.save(temp_file.name)
        filename = temp_file.name

    try:
        pygame.mixer.quit()
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.5)

    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        for _ in range(10):
            try:
                os.remove(filename)
                break
            except PermissionError:
                time.sleep(0.2)

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSI")
        self.geometry("800x500")

        self.paused = tk.BooleanVar(value=False)
        self.exited = tk.BooleanVar(value=False)
        
        self._create_widgets()
        self._layout_widgets()

        self.console_left.tag_config("bold_red", foreground="red", font=("bold"))
        self.console_left.tag_config("bold_blue", foreground="blue", font=("bold"))

        self.console_right.tag_config("Emote", foreground="red", wrap=None)
        self.console_right.tag_config("Joint", foreground="blue", wrap=None)
        self.console_right.tag_config("Action", font=("bold"))


    def _create_widgets(self):
        # Row 1: Title Label
        self.title_label = tk.Label(self, text="CSI SAMI CONSOLE", font=("Arial", 16, "bold"))

        # Row 2: Dropdown menu
        self.dropdown_var = tk.StringVar()
        self.dropdown_menu = ttk.Combobox(self, textvariable=self.dropdown_var, state="readonly")
        self.dropdown_menu['values'] = mic_names
        self.dropdown_menu.current(0)

        self.status_frame = tk.Frame(self)
        self.button = tk.Button(self.status_frame, text="EXIT", command=self.exit_game)
        self.switch = ttk.Checkbutton(self.status_frame, text="Pause", variable=self.paused, command=self.toggle_pause)

        # Row 3: Two side-by-side consoles
        self.console_left = ScrolledText(self, wrap=tk.WORD, width=50, height=20, bg="#FDFDFD", font=("Segoe UI", 12))
        self.console_right = ScrolledText(self, wrap=None, width=50, height=20, bg="#FDFDFD", font=("Segoe UI", 12))

        # Example: insert starter text
        self.console_left.insert(tk.END, "Dialogue Log...\n")
        self.console_right.insert(tk.END, "Console Log...\n")

    def insert_console_left(self, message):
        self.console_left.insert(tk.END, time.strftime("%H:%M:%S ", time.localtime(time.time())), "bold")
        if message[0] == "EXIT":
            self.exit_game()
        if message[0] == "LLM":
            self.console_left.insert(tk.END, "SAMI says: ", "bold_red")
            self.console_left.insert(tk.END, message[1] + '\n')
        elif message[0] == "user":
            self.console_left.insert(tk.END, "You say: ", "bold_blue")
            self.console_left.insert(tk.END, message[1] + '\n')
        else:
            pass
        self.console_left.see('end')

    def insert_console_right(self, message):
        self.console_right.insert(tk.END, time.strftime("%H:%M:%S ", time.localtime(time.time())), "bold")
        if message[0] == "Emote" or message[0] == "Joint":
            self.console_right.insert(tk.END, "Sent " + message[0] + " command: ", message[0])
            self.console_right.insert(tk.END, message[1] + '\n')
        if message[0] == "Action":
            self.console_right.insert(tk.END, message[1] + '\n', message[0])
        else:
            pass
        self.console_right.see('end')

    def _layout_widgets(self):
        # Grid layout with uniform rows
        self.grid_rowconfigure(2, weight=1)  # row 3 stretches vertically
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)
        self.dropdown_menu.grid(row=1, column=0, pady=5, sticky="ew", padx=20)

        self.status_frame.grid(row=1, column=1)
        self.switch.pack(side="left")
        self.button.pack(side="right")

        self.console_left.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="nsew")
        self.console_right.grid(row=2, column=1, padx=(10, 20), pady=10, sticky="nsew")

    def check_queue(self):
        try:
            while True:
                self.insert_console_left(qL.get_nowait())
        except Empty:
            pass
        try:
            while True:
                self.insert_console_right(qR.get_nowait())
        except Empty:
            pass
        self.after(50, self.check_queue)

    def toggle_pause(self):
        self.pause = self.paused.get()
    
    def exit_game(self):
        self.destroy()

########################################################################
# save game to text file
########################################################################
# would you like to save your conversation - will have to change the save_data 
#also probably help ful for intro project 
def save_game():
    text_to_speech("Would you like to save your game?")
    
    #quick check for text only mode
    if TEXT_ONLY_MODE:
        response = speech_to_text()
    else:
        response = speech_to_text(mic_index)

    while True:
        if response in {"yes", "y"}:
            save_data = {
                "inventory": inventory, 
                "full_story": full_story,
                "recent_turns": recent_turns,
                "story_summary": story_summary,
            }
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4)
                return

        elif response in {'n', 'no'}:
            return 
        else:
            print("Please enter 'yes' or 'no'!") 

########################################################################
# if save exists, load it
########################################################################
#if prior model 
def load_game():
    global inventory, full_story, recent_turns, story_summary
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            inventory = data.get("inventory", {})
            full_story = data.get("full_story", "")
            recent_turns = data.get("recent_turns", [])
            story_summary = data.get("story_summary", "")
            return data.get("context", "")
    return None

########################################################################
# detects if a previous save file exists
########################################################################
def load_state():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return None

########################################################################
# prompt user to choose new or saved game
########################################################################

#would have to add additional logic check for if known person... for intro project + person 
def load_saved_game():
    while True:
        user_choice = input("Previous game file found. Would you like to continue? (y/n): ").strip().lower()
        if user_choice in {'y', 'yes'}:
            print("loading saved game... \n")
            return True
        elif user_choice in {'n', 'no'}:
            print ("Beginning new game... \n")
            return False
        else:
            print("Please enter 'yes' or 'no'!")

########################################################################
# Begin the game
########################################################################
def init_game():
    continue_game = False
    loaded = load_state()        #load game only returns true or Null... can remove loaded I think

    if loaded:
        continue_game = load_saved_game()

    if continue_game:
        global story_summary, recent_turns
        load_game()
        print(f"The game so far: {story_summary}\nMost Recent move: {recent_turns[2]}")
    else:
        pass
        #text_to_speech("Welcome to The Inn, a Collaborative Storytelling Setting. Say begin to start game. Say quit to exit.")

########################################################################
# update inventory
########################################################################
def update_inventory(query):
    query = query.lower()

    # Check inventory
    if query in {"inventory", "check inventory", "show inventory"}:
        if not inventory:
            text_to_speech("Your inventory is empty.")
        else:
            inv_list = ', '.join(f"{item} (x{count})" for item, count in inventory.items())
            text_to_speech(f"You have: {inv_list}")
        return False

    # Pick up or take item
    match = re.search(r"(pick up|take)\s+(?:a|an|the)?\s*(.+)", query)
    if match:
        item = match.group(2).strip()
        inventory[item] = inventory.get(item, 0) + 1
        text_to_speech(f"You picked up a {item}.")
        return True

    # Use item
    match = re.search(r"use\s+(?:a|an|the)?\s*(.+)", query)
    if match:
        item = match.group(1).strip()
        if item in inventory and inventory[item] > 0:
            inventory[item] -= 1
            if inventory[item] == 0:
                del inventory[item]
            text_to_speech(f"You use the {item}.")
        else:
            text_to_speech(f"You don't have a {item} to use.")
        return True

    # Drop or remove item
    match = re.search(r"(drop|remove)\s+(?:a|an|the)?\s*(.+)", query)
    if match:
        item = match.group(2).strip()
        if item in inventory:
            inventory[item] -= 1
            if inventory[item] <= 0:
                del inventory[item]
            text_to_speech(f"You dropped the {item}.")
        else:
            text_to_speech(f"You don't have a {item}.")
        return True

    return True

########################################################################
# format inventory
########################################################################
def format_inventory_for_prompt():
    if not inventory:
        return "The player's inventory is empty."
    else:
        return "The player has the following items: " + ", ".join(
            f"{item} (x{count})" for item, count in inventory.items()
        )

########################################################################
# function that builds the prompt to send to the LLM for it's DM response
########################################################################
def construct_prompt(query):
    prompt = f"summary of story so far:\n{story_summary}\n\n"
    prompt += f"recent interactions:\n" + "\n".join(recent_turns) + "\n"
    prompt += f"Player: {query}\nNarator: "
    return prompt

########################################################################
# function that summarizes older interactions to keep prompts short
########################################################################
def summarize_story(oldest, story_summary):
    prompt = f"""create a summary of the interactions below. Be sure to keep all important story beats and player interactions. 
            The summary should be coherent and anybody who reads it should be able to read it and remember what came before\n
            current summary: {story_summary}\n
            Newest interaction: {oldest}"""
    return transmit_prompt(prompt)

def gameloop(exit_flag, paused):
    global recent_turns, story_summary, full_story
    while not exit_flag.is_set():
        ser = initialize_serial_connection()
        
        while paused.get():
            pass

        if not exit_flag.is_set():
            stop_event = Event()
            emote_thread = threading.Thread(target=read_json, args=(ser, "config/Listening.json", stop_event))
            emote_thread.start()
            query = speech_to_text(mic_index)
            stop_event.set()
            time.sleep(0.05)
            emote_thread.join()

            print('\033[F' + '\033[1m' + "You say: " + '\033[0m' + query + '\033[K' + '\n', end='\033[K')
            qL.put(["user", query])

        while paused.get():
            pass
            
        # check for quit or save
        if query in {"quit", "exit"}:
            read_json(ser, "config/home.json", Event())
            time.sleep(1.05)
            qL.put(["EXIT", None])
            return
        elif query in {"save", "save game"}:
            save_game()
            read_json(ser, "config/home.json", Event())
            time.sleep(1.05)
            close_connection(ser)
            qL.put(["EXIT", None])
            return
        # call inventory func. skip LLM if all player did was check inventory or use nonexistent item.
        if update_inventory(query) is False:
            read_json(ser, "config/home.json", Event())
            time.sleep(1.05)
            close_connection(ser)
            continue

        if not exit_flag.is_set():
            stop_event = Event()
            emote_thread = threading.Thread(target=read_json, args=(ser, "config/Thinking.json", stop_event))
            emote_thread.start()
            prompt = construct_prompt(query)
            response = transmit_prompt(prompt)
            stop_event.set()
            time.sleep(0.05)
            emote_thread.join()

            print('\033[1m' + "SAMI says: " + '\033[0m' + response + '\033[K')
            qL.put(["LLM", response])

        while paused.get():
            pass

        if not exit_flag.is_set():
            stop_event = Event()
            emote_thread = threading.Thread(target=read_json, args=(ser, get_gesture(response), stop_event))
            emote_thread.start()
            text_to_speech(response)        
            stop_event.set()
            time.sleep(0.05)
            emote_thread.join()
        read_json(ser, "config/home.json", Event())

        #save the appropriate bits of information in the correct spaces. and summarize where appropriate
        latest_interaction = f"Player: {query}\nNarrator:{response}"
        recent_turns.append(latest_interaction)

        #keep the most_recent list to 3 interactions and summarize everything before that
        if len(recent_turns) > 3:
            oldest = recent_turns.pop(0)
            story_summary = summarize_story(oldest, story_summary)
        
        # add messages to full story
        full_story += f"Player: {query}\nNarrator:{response}"

        time.sleep(1.05)

        close_connection(ser)
        ser = None


def main():
    init_game()
    exit_flag = Event()

    #create gui
    app = GUI()

    #initalzing threads - need to better understand gameloop - seems to be main function of interaction
    MUD_thread = threading.Thread(target=gameloop,args=(exit_flag,app.paused))

    #start thread
    MUD_thread.start()

    #for the gui - what is mainloop - builtin function?
    try:
        app.after(50, app.check_queue)
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        print('\x1B[3m' + "Exited by user." + '\x1B[0m' + '\033[K')
        qR.put(["Action", "Exited by user."])
        exit_flag.set()
        MUD_thread.join()


if __name__ == "__main__":
  main()
