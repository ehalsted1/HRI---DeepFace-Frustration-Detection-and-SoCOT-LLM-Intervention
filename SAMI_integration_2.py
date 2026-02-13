#IMPORTS from absolute path
import importlib.util
import sys

#sami imports
from SAMI_Robot.software.audio_manager import AudioManager
from SAMI_Robot.software.SAMIControl import SAMIControl
from SAMI_Robot.software import SAMI_UI

#to be able to access serial ports
import serial

#threading shares memory is ideal for IO (serial, api calls, waiting)
import threading 

#to be able to keep time?
import time

#using queue: LLM determine SAMI actions
from queue import Queue 
from queue import Empty

import random
import os

from llm import Conversation, LLM_State, InteractionState
import logging
from logging_config import init_logging, resolve_debug_flag

from ui import LLM_UI

# llm: sends (initial) command (sets)
# sami: recieves official command (unset)
# sami: performs behavior (set)
# llm: Wait until set (sami performed behavior) to send next command 

# by checking import we run them 

#config
# ARDUINO_PORT = '/dev/cu.usbserial-10'

# OR 
ARDUINO_PORT = '/dev/cu.usbserial-110'

from enum import Enum, auto

#test functions for sending 

def llm_task(command_queue, stop_event, robot_busy):
    """
    Simulates receiving robot commands from an LLM and sending them to the robot.
    """
    import random
    possible_commands = {"audio_question": ["thinking.json", "empatheticQuestion.json"], "greeting": ["wave.json"], "recording_listening":["nodding.json"]}

    while not stop_event.is_set():

        if stop_event.is_set():
            break

        if robot_busy.is_set():
            time.sleep(1)
            continue
        for _ in range(100):
            if stop_event.is_set():
                return
            time.sleep(0.1)
        # simulate thinking / waiting for LLM
        command = random.choice(list(possible_commands.items()))
        command_type = command[0]
        command_json = random.choice(command[1])

        print(f"[LLM] generated command: ", command_type)
        time.sleep(10)

         # Send command to robot
        command_queue.put(command_json)

        # print(f"Command type: %s, command val: %s" % (command_type, command_json))
        print(f"[LLM] sent command: {command_json}")

def llm_task_app(command_queue, stop_event, robot_busy, interaction_state):
    state_to_command = {LLM_State.THINKING: "thinking.json", 
                        LLM_State.SPEAKING: "empatheticQuestion.json", 
                        LLM_State.LISTENING: "nodding.json"}
    
    last_state = None
    pending_state = None

    while not stop_event.is_set():
        
        if robot_busy.is_set():
            logging.debug("Busy SAMI")
            time.sleep(0.5)
            continue
        
        if pending_state is None:
            try: 
                pending_state = interaction_state.get()
            except Empty:
                print("Empty state")
                continue

        state = pending_state
        logging.debug("[LLM State]: ", state)

        if state == LLM_State.SHUTDOWN:
            stop_event.set()
            break

        if state != last_state and state in state_to_command:
            cmd = state_to_command[state]
            command_queue.put(cmd)
            logging.debug("[LLM State]: ", state)
            print(f"state: ", state)
            logging.debug("LLM Command Added: ", cmd)
            logging.debug("Num Queue after LLM command added: ", command_queue.qsize())
            last_state = state

        pending_state = None 
    

def robot_task(command_queue, stop_event, robot_busy):
    """
    Simulates a robot executing commands received from the LLM.
    Replace print() calls with actual Arduino serial writes.
    """
    robot = SAMIControl(arduino_port=ARDUINO_PORT, audio_folder="audio", starting_voice="Matt")
    robot.initialize_serial_connection()
    print(f"[SAMI] Robot intialized.")

    robot.run_behavior_block("Home.json")
    # robot.run_behavior_block("Wave.json")
    # robot.run_behavior_block("Home.json")
    command = None

    try:
        while not stop_event.is_set():
            try:
                #block until command in queue
                command = command_queue.get(timeout=1)
                print(f"Command Recieved for Robot: %s", command)
                print(f"Num Queue after command received: %d", command_queue.qsize())
            except Empty:
                continue
        
            robot_busy.set()
            print(f"[SAMI] Executing command: {command}")

            try:
                logging.info("[SAMI] Executing command: %s", command)
                robot.run_behavior_block(command)
                logging.info("[SAMI] Finished command: %s", command)
                robot.run_behavior_block("Home.json")

            except Exception:
                logging.exception("[SAMI] Error Executing: %s", command)
            finally:
                #sami is now able to take commands 
                robot_busy.clear()           
                #mark command done
                command_queue.task_done()
    except Exception as e:
        logging.debug("error %s", e)
    
    finally: 
        robot.run_behavior_block("Home.json")
        robot.close_connection()
        print(f"[SAMI] connection closed")


def main():
    command_queue = Queue()
    stop_event = threading.Event()
    robot_busy = threading.Event()

    interaction_state = InteractionState()

    DB_PATH = os.path.join(os.getcwd(), "database_llm")
    conversation_path = os.path.join(DB_PATH)
    convo = Conversation(conversation_path=conversation_path, interaction_state=interaction_state)


    #make our threads for sami + LLM
    t1_sami = threading.Thread(target=robot_task, args=(command_queue, stop_event, robot_busy))
    t2_llm = threading.Thread(target=llm_task_app, args=(command_queue, stop_event, robot_busy, interaction_state), daemon=True)

    #start threads
    t2_llm.start()
    t1_sami.start()

    print("Press 'ctrl+c' to quit.")
    

    try:
        print("Starting conversation")
        init_logging(convo.debug)
        response = convo.run_ollama()
        convo.logger.info("[LLM]: %s", response)
        convo.text_to_speech(response)
        
        print("recording")
        convo.logger.info("Conversation loop starting for %s (debug=%s).", convo.conversation_name, convo.debug)
        while True:
            convo.stt_result = convo.record_and_transcribe_until_silence()
            if convo.stt_result == "" or convo.stt_result == " ":
                convo.logger.warning("No response detected from user during conversation.")
                convo.engine.say("Sorry I couldn't catch that, could you please say that again?")
                convo.engine.runAndWait()
                continue
            if "done" in convo.stt_result.lower() or "quit" in convo.stt_result.lower() or "end conv" in convo.stt_result.lower() or "and conv" in convo.stt_result.lower():
                convo.engine.say("It was great talking to you. Have a good day.")
                convo.logger.info("User requested to end the conversation.")
                break
            convo.conversation.append({'role': 'user', 'content':convo.stt_result})
            convo.logger.info("[User]: %s", convo.stt_result)
            
            responding = convo.run_ollama()
            convo.logger.info("[LLM]: %s", responding)

            convo.text_to_speech(responding)
            convo.logger.info("[LLM]: Responded TTS")

            convo.remember_conversation()

        if convo.interaction_state:
            convo.interaction_state.set(LLM_State.SHUTDOWN)
        convo.logger.info("Conversation loop concluded for %s.", convo.conversation_name)

    except KeyboardInterrupt:
        print("Main: KeyboardInterrupt detected")
        stop_event.set()
        

    #why do we have to join them?
    t1_sami.join()
    t2_llm.join()

    print("Main: stopped all threads")



if __name__ == "__main__":
    main()