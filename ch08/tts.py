# 1 First, we must install pyttsx for the Python 3 library (offline TTS for Python 3) and its relative dependencies:
# pip install pyttsx3

# 2 To avoid possible errors, it is also necessary to install the pypiwin32 library
# pip install pypiwin32

# 3 Let's import the pyttsx3 package
import pyttsx3

# 4 We create an engine instance that will use the specified driver:
engine = pyttsx3.init()

# 5 To change the speech rate, use the following commands:
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-50)

# 6 To change the voice of the speaker, use the following commands:
voices = engine.getProperty('voices')
engine.setProperty('voice', 'TTS_MS_EN-US_ZIRA_11.0')

# 7 Now, we will use the say method to queue a command to speak an utterance.
# The speech is output according 
# to the properties set before this command in the queue:
engine.say("You are reading the Python Machine Learning Cookbook")
engine.say("I hope you like it.")
engine.say("My name is Eebrahim Ouedraogo")

# 8 Finally, we will invoke the runAndWait() method. This method blocks while 
# processing all currently queued commands and invokes callbacks for engine 
# notifications appropriately. It returns when all commands queued before this 
# call are emptied from the queue:
engine.runAndWait()

# At this point, a different voice will read the text supplied by us.