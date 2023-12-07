#!/usr/bin/python3

import pyxhook
import time
import os 
from RecordKeyForLogin import RecordKeyForLogin
from Keystroke_Dynamics_eval import KeystrokeDynamicsEvaluator


# Create CaptureTime object
capTime = RecordKeyForLogin()

def banner():
	print("""
                                                                               
 _____             _           _             ____                    _         
|  |  |___ _ _ ___| |_ ___ ___| |_ ___      |    \ _ _ ___ ___ _____|_|___ ___ 
|    -| -_| | |_ -|  _|  _| . | '_| -_|     |  |  | | |   | .'|     | |  _|_ -|
|__|__|___|_  |___|_| |_| |___|_,_|___|_____|____/|_  |_|_|__,|_|_|_|_|___|___|
          |___|                       |_____|     |___|                        
	""")


user = input("\n[-] Enter the username: ")
capTime.setUser(user)

evaluator = KeystrokeDynamicsEvaluator()

# Set default values for repetitions and session
capTime.setSessionFreq(1)
capTime.setEntryFreq(1)

# Create dictionary which will be used to store raw times
capTime.CreateDicTimes()

# Create hookmanager
hookman = pyxhook.HookManager()
# Define our callback to fire when a key is pressed down
hookman.KeyDown = capTime.KeyDownEvent
hookman.KeyUp = capTime.KeyUpEvent
# Hook the keyboard
hookman.HookKeyboard()
# Start our listener
hookman.start()

# Start capturing keystrokes
capTime.CalculateKeystrokesDynamics()

# Create a loop to keep the application running
while capTime.running:
    time.sleep(0.1)

# Close the listener when we are done
hookman.cancel()

training_data_path = "output/user_timings.csv"
testing_data_path = "output/Predict_keystrokes.csv"
evaluator.user_accept(user)
evaluator.load_data(training_data_path, testing_data_path)
