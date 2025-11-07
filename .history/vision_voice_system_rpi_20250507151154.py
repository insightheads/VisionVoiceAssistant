import os
import cv2
import face_recognition
import numpy as np
from fer import FER
import mediapipe as mp
from ultralytics import YOLO
import time
import speech_recognition as sr
import pyttsx3
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf
import string
import json
import wikipedia
import threading
from queue import Queue
import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import PyPDF2
import docx
from pptx import Presentation
import warnings
from plyer import notification
import re
import dateparser
import geocoder
import folium
import webbrowser
from pathlib import Path
import math
from word2number import w2n
import requests
from bs4 import BeautifulSoup
import urllib.parse
import _thread

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure display backend for Raspberry Pi
try:
    # Try X11 first
    os.environ['DISPLAY'] = ':0'
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
except Exception as e:
    print(f"Warning: Could not set display backend: {str(e)}")

# Function to create window with proper backend
def create_window(window_name):
    """Create a window with proper backend handling."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Try to set window properties
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    except Exception as e:
        print(f"Warning: Could not create window with normal mode: {str(e)}")
        try:
            # Fallback to basic window
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"Error: Could not create window: {str(e)}")
            return None
    return window_name

# Initialize voice components
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 1.0)

# Global variables for speech control
speech_in_progress = False
stop_speech = False
speech_lock = threading.Lock()
continuous_listening_enabled = True

# Try to set up event handlers for speech interruption
try:
    engine.connect('started-word', lambda name, location, length: check_for_interruption())
except Exception as e:
    print(f"Warning: Could not connect speech event handlers: {str(e)}")

def check_for_interruption():
    """Check if speech should be interrupted."""
    global stop_speech
    if stop_speech:
        try:
            engine.stop()
            print("Speech interrupted")
        except Exception as e:
            print(f"Error stopping speech: {str(e)}")

# Performance optimization: Set OpenCV backend
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL backend enabled")
else:
    print("OpenCL not available, using default backend")

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30
MAX_CAMERA_RETRIES = 3
CAMERA_RETRY_DELAY = 2  # seconds

# Camera device indices for Raspberry Pi
CAMERA_DEVICES = [
    0,      # Default V4L2 device
    2,      # Common USB camera index
    '/dev/video0',  # Direct V4L2 device path
    -1      # Default camera
]

# Constants for distance estimation
KNOWN_WIDTH = 15.0
FOCAL_LENGTH = 600
KNOWN_OBJECT_WIDTH = 180.0

# Initialize models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_detector = FER(mtcnn=False)
back_sub = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)

# Initialize YOLO
model_yolo = YOLO("yolov8n.pt")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Known faces storage
known_face_encodings = []
known_face_names = []

# Knowledge base setup
knowledge_base_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base.json")
knowledge_base = {}  # Global variable for knowledge base

# Reminders and alarms storage
reminders_file = "reminders.json"
reminders = []  # List to store reminders
active_timers = []  # List to store active timers

# Gesture library
gesture_library = {
    "Hello": [0, 1, 1, 1, 1],
    "Thank You": [1, 0, 0, 0, 0],
    "Yes": [1, 0, 0, 0, 0],
    "No": [1, 1, 0, 0, 0],
    "Help": [1, 0, 0, 0, 1],
    "Stop": [0, 1, 1, 1, 1]
}

def initialize_knowledge_base():
    """Initialize the knowledge base with default structure and values."""
    global knowledge_base
    knowledge_base = {
        "general": {
            "robot_name": "Vision Voice Assistant",
            "version": "1.0",
            "created_date": datetime.datetime.now().strftime("%Y-%m-%d")
        },
        "commands": {
            "hello": "Greeting recognized. Hello there!",
            "how are you": "I'm functioning normally. Thank you for asking.",
            "what time is it": "I can tell you the current time.",
            "what is your name": "I am the Vision Voice Assistant, your helpful robot companion."
        },
        "facts": {},
        "weather": {},
        "news": {},
        "locations": {
            "home": {
                "latitude": 0,
                "longitude": 0,
                "description": "Home base location"
            }
        },
        "preferences": {
            "voice_rate": 175,
            "voice_volume": 1.0,
            "detection_confidence": 0.7
        },
        "user_data": {
            "name": "User",
            "face_id": None,
            "preferences": {
                "greeting": "Hello User, how can I help you today?"
            }
        }
    }
    print("Knowledge base initialized with default structure")
    return knowledge_base

def load_knowledge_base():
    """Load knowledge base from JSON file with better error handling."""
    global knowledge_base
    try:
        print(f"Attempting to load knowledge base from: {knowledge_base_file}")
        if os.path.exists(knowledge_base_file):
            with open(knowledge_base_file, 'r', encoding='utf-8') as f:
                try:
                    knowledge_base = json.load(f)
                    print("Knowledge base loaded successfully")
                    print(f"Current entries: {len(knowledge_base)} items")
                except json.JSONDecodeError as jde:
                    print(f"Error decoding knowledge base file: {str(jde)}")
                    print("Creating new knowledge base.")
                    initialize_knowledge_base()
                    save_knowledge_base()
        else:
            print(f"No existing knowledge base found at {knowledge_base_file}")
            print("Creating new knowledge base.")
            initialize_knowledge_base()
            save_knowledge_base()
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        initialize_knowledge_base()
        save_knowledge_base()

def save_knowledge_base():
    """Save knowledge base to JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(knowledge_base_file), exist_ok=True)
        
        with open(knowledge_base_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=4, ensure_ascii=False)
        print(f"Knowledge base saved successfully to {knowledge_base_file}")
    except Exception as e:
        print(f"Error saving knowledge base: {str(e)}")

def add_to_knowledge_base(category, key, value):
    """Add a new entry to the knowledge base.
    
    Args:
        category: The category in the knowledge base (e.g., 'general', 'commands', 'facts')
        key: The key within the category
        value: The value to store
    """
    global knowledge_base
    try:
        # Create category if it doesn't exist
        if category not in knowledge_base:
            knowledge_base[category] = {}
        
        # Add or update the entry
        knowledge_base[category][key] = value
        save_knowledge_base()
        print(f"Added new entry to knowledge base: {category}.{key}")
        return True
    except Exception as e:
        print(f"Error adding to knowledge base: {str(e)}")
        return False

def get_from_knowledge_base(category, key=None):
    """Retrieve a value from the knowledge base.
    
    Args:
        category: The category in the knowledge base (e.g., 'general', 'commands', 'facts')
        key: The key within the category. If None, returns the entire category.
    
    Returns:
        The value if found, None otherwise.
    """
    global knowledge_base
    try:
        if category not in knowledge_base:
            return None
            
        if key is None:
            return knowledge_base.get(category, {})
        
        return knowledge_base.get(category, {}).get(key, None)
    except Exception as e:
        print(f"Error retrieving from knowledge base: {str(e)}")
        return None

def load_known_faces(known_faces_dir='known_faces'):
    """Load known faces from a directory."""
    if not os.path.exists(known_faces_dir):
        print(f"Warning: {known_faces_dir} directory not found")
        return
        
    for filename in os.listdir(known_faces_dir):
        if filename.endswith((".jpg", ".png")):
            try:
                image_path = os.path.join(known_faces_dir, filename)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encoding = face_recognition.face_encodings(rgb_image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"Loaded face: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

def recognize_gesture(landmarks):
    """Recognize a hand gesture."""
    fingers = []
    # Thumb
    fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    # Other fingers
    for i in [8, 12, 16, 20]:
        fingers.append(1 if landmarks[i].y < landmarks[i - 2].y else 0)
    
    for gesture, pattern in gesture_library.items():
        if fingers == pattern:
            return gesture
    return None

def speech_to_text(timeout=5, phrase_time_limit=5):
    """Convert speech to text."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.WaitTimeoutError:
        print("No speech detected within timeout")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None
    except Exception as e:
        print(f"Error in speech recognition: {str(e)}")
        return None

def non_blocking_listen(command_queue):
    """Continuously listen for user input in a non-blocking way."""
    global speech_in_progress, stop_speech
    
    while True:
        try:
            # Use a shorter timeout for more responsive interruption
            command = speech_to_text(timeout=3, phrase_time_limit=5)
            
            if command:
                command = command.lower()
                
                # Check for STOP command to interrupt speech
                if "stop" in command and speech_in_progress:
                    print("STOP command detected - interrupting speech")
                    with speech_lock:
                        stop_speech = True
                    continue
                
                # If speech is in progress, interrupt it for the new command
                if speech_in_progress:
                    print("New command detected while speaking - interrupting speech")
                    with speech_lock:
                        stop_speech = True
                    # Small delay to allow speech to stop
                    time.sleep(0.2)
                
                # Process the command
                command_queue.put(('VOICE_INPUT', command))
        except Exception as e:
            print(f"Error in continuous listening: {str(e)}")
            time.sleep(1)  # Prevent tight loop in case of repeated errors

def text_to_speech(text):
    """Convert text to speech with interruption capability."""
    global speech_in_progress, stop_speech
    
    try:
        # Set speech in progress flag
        with speech_lock:
            speech_in_progress = True
            stop_speech = False
        
        print(f"Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Break text into smaller chunks for more responsive interruption
        sentences = text.split('.')
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Check if we should stop before saying the next sentence
            with speech_lock:
                if stop_speech:
                    print("Speech interrupted between sentences")
                    break
            
            # Say this sentence
            engine.say(sentence + '.')
            engine.runAndWait()
            
            # Brief pause to check for interruptions
            time.sleep(0.1)
            
            # Check again if we should stop
            with speech_lock:
                if stop_speech:
                    print("Speech interrupted between sentences")
                    break
        
        # Reset speech flags when done
        with speech_lock:
            speech_in_progress = False
            stop_speech = False
            
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        # Make sure to reset flags even on error
        with speech_lock:
            speech_in_progress = False
            stop_speech = False

def continuous_listen(command_queue):
    """Continuously listen for user input in a non-blocking way."""
    global speech_in_progress, stop_speech, continuous_listening_enabled
    
    print("Starting continuous listening thread...")
    
    # Create a dedicated recognizer for this thread
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000  # Lower threshold for better sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5  # Shorter pause threshold for quicker response
    
    while continuous_listening_enabled:
        try:
            # Use a shorter timeout for more responsive interruption
            with sr.Microphone() as source:
                if not speech_in_progress:
                    print("Continuous listener ready...")
                # Shorter adjustment for ambient noise to be more responsive
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                
                try:
                    # Use a shorter timeout to be more responsive
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Process the audio
                    text = recognizer.recognize_google(audio)
                    command = text.lower()
                    print(f"Continuous listener heard: {command}")
                    
                    # Check for STOP command to interrupt speech
                    if any(word in command for word in ["stop", "shut up", "be quiet", "silence"]) and speech_in_progress:
                        print("STOP command detected - interrupting speech")
                        with speech_lock:
                            stop_speech = True
                            # Immediately reset speech_in_progress so new commands can be processed
                            speech_in_progress = False
                        
                        # Force stop the engine
                        try:
                            engine.stop()
                        except Exception as e:
                            print(f"Error stopping engine: {str(e)}")
                            
                        # Print acknowledgment that we stopped
                        print("Speech stopped, ready for new commands")
                        
                        # Don't continue - allow processing of the next command
                    
                    # If speech is in progress, interrupt it for the new command
                    if speech_in_progress:
                        print("New command detected while speaking - interrupting speech")
                        with speech_lock:
                            stop_speech = True
                            
                        # Force stop the engine
                        try:
                            engine.stop()
                        except Exception as e:
                            print(f"Error stopping engine: {str(e)}")
                            
                        # Small delay to allow speech to stop
                        time.sleep(0.2)
                    
                    # Process the command
                    command_queue.put(('VOICE_INPUT', command))
                    
                except sr.WaitTimeoutError:
                    # This is expected, just continue listening
                    pass
                except sr.UnknownValueError:
                    # Could not understand audio, continue listening
                    pass
                except Exception as e:
                    print(f"Error in continuous listening recognition: {str(e)}")
        except Exception as e:
            print(f"Error in continuous listening: {str(e)}")
            time.sleep(0.5)  # Shorter delay to be more responsive

def init_camera():
    """Initialize camera with retries and multiple device options."""
    print("Initializing camera...")
    
    for retry in range(MAX_CAMERA_RETRIES):
        for device in CAMERA_DEVICES:
            try:
                if isinstance(device, str):
                    # For direct device path
                    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                else:
                    cap = cv2.VideoCapture(device)
                
                if cap.isOpened():
                    # Configure camera settings
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                    
                    # Verify settings were applied
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    print(f"Camera initialized successfully:")
                    print(f"- Device: {device}")
                    print(f"- Resolution: {actual_width}x{actual_height}")
                    print(f"- FPS: {actual_fps}")
                    return cap
                else:
                    print(f"Failed to open camera device: {device}")
                    cap.release()
            except Exception as e:
                print(f"Error initializing camera device {device}: {str(e)}")
                if 'cap' in locals() and cap is not None:
                    cap.release()
        
        if retry < MAX_CAMERA_RETRIES - 1:
            print(f"Retrying camera initialization in {CAMERA_RETRY_DELAY} seconds...")
            time.sleep(CAMERA_RETRY_DELAY)
    
    raise RuntimeError("Failed to initialize camera after multiple attempts")

def search_wikipedia(query):
    """Search Wikipedia with better error handling and fallback options."""
    try:
        print(f"Searching Wikipedia for: {query}")
        
        # First try direct Wikipedia search with better error handling
        try:
            result = wikipedia.summary(query, sentences=3)
            print("Wikipedia search successful")
            return result, True
        except wikipedia.DisambiguationError as e:
            # If disambiguation page, try the first option
            print(f"Disambiguation error. Options: {e.options[:3]}")
            try:
                result = wikipedia.summary(e.options[0], sentences=3)
                return f"Showing results for '{e.options[0]}': {result}", True
            except Exception as inner_e:
                print(f"Error with disambiguation option: {str(inner_e)}")
        except wikipedia.PageError:
            print(f"Page not found for: {query}")
        except Exception as e:
            print(f"Wikipedia error: {str(e)}")
        
        # Try search suggestion as fallback
        try:
            print("Trying search suggestions...")
            suggestions = wikipedia.search(query, results=3)
            if suggestions:
                print(f"Found suggestions: {suggestions}")
                for suggestion in suggestions:
                    try:
                        result = wikipedia.summary(suggestion, sentences=3)
                        return f"Showing results for '{suggestion}': {result}", True
                    except:
                        continue
        except Exception as e:
            print(f"Error with search suggestions: {str(e)}")
        
        # If all else fails, try a more general search
        try:
            modified_query = query.split()[0] if len(query.split()) > 1 else query
            print(f"Trying more general search with: {modified_query}")
            result = wikipedia.summary(modified_query, sentences=2)
            return f"Couldn't find exact match, but here's information about '{modified_query}': {result}", True
        except Exception as e:
            print(f"General search failed: {str(e)}")
        
        return f"Sorry, I couldn't find information about '{query}' on Wikipedia.", False
    except Exception as e:
        print(f"Wikipedia search error: {str(e)}")
        return f"Sorry, I encountered an error while searching for '{query}'.", False

def load_reminders():
    """Load reminders from JSON file."""
    global reminders
    try:
        if os.path.exists(reminders_file):
            with open(reminders_file, 'r', encoding='utf-8') as f:
                reminders = json.load(f)
                print("Reminders loaded successfully")
        else:
            print("No existing reminders found. Creating new file.")
            reminders = []
            save_reminders()
    except Exception as e:
        print(f"Error loading reminders: {str(e)}")
        reminders = []

def save_reminders():
    """Save reminders to JSON file."""
    try:
        with open(reminders_file, 'w', encoding='utf-8') as f:
            json.dump(reminders, f, indent=4)
        print("Reminders saved successfully")
    except Exception as e:
        print(f"Error saving reminders: {str(e)}")

def parse_datetime_string(datetime_str):
    """Parse datetime string using dateparser library."""
    try:
        # First try dateparser
        parsed_date = dateparser.parse(datetime_str, settings={
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': datetime.datetime.now()
        })
        
        if parsed_date:
            return parsed_date
            
        # If dateparser fails, try our custom parser
        current_date = datetime.datetime.now()
        
        # Clean up the input string
        datetime_str = datetime_str.lower().strip()
        datetime_str = datetime_str.replace(" at ", " ").replace(" by ", " ")
        
        # Try to parse with custom formats
        formats = [
            "%d/%m %I:%M %p",
            "%d/%m %I:%M%p",
            "%d/%m %I %p",
            "%d/%m %I%p",
            "%Y-%m-%d %H:%M",
        ]
        
        for fmt in formats:
            try:
                parsed_date = datetime.datetime.strptime(datetime_str, fmt)
                if "%Y" not in fmt:
                    parsed_date = parsed_date.replace(year=current_date.year)
                    if parsed_date < current_date:
                        parsed_date = parsed_date.replace(year=current_date.year + 1)
                return parsed_date
            except ValueError:
                continue
        
        raise ValueError("Could not parse date")
    except Exception as e:
        raise ValueError(f"Could not understand the date/time format. Please try again.")

def add_reminder(title, datetime_str, description=""):
    """Add a new reminder."""
    try:
        print(f"\nTrying to add reminder with date: {datetime_str}")
        reminder_time = parse_datetime_string(datetime_str)
        
        reminder = {
            "title": title,
            "time": reminder_time.strftime("%Y-%m-%d %H:%M"),
            "description": description,
            "completed": False
        }
        reminders.append(reminder)
        save_reminders()
        text_to_speech(f"Reminder set for {title} at {reminder_time.strftime('%d %B at %I:%M %p')}")
        return True
    except ValueError as e:
        text_to_speech(str(e))
        return False

def start_timer(duration_minutes, title="Timer"):
    """Start a new timer."""
    try:
        end_time = datetime.datetime.now() + datetime.timedelta(minutes=duration_minutes)
        timer = {
            "title": title,
            "end_time": end_time,
            "duration": duration_minutes
        }
        active_timers.append(timer)
        text_to_speech(f"Timer set for {duration_minutes} minutes")
        return True
    except Exception as e:
        text_to_speech("Failed to set timer. Please try again.")
        return False

def check_reminders():
    """Check for due reminders and notify user."""
    current_time = datetime.datetime.now()
    for reminder in reminders:
        if not reminder["completed"]:
            reminder_time = datetime.datetime.strptime(reminder["time"], "%Y-%m-%d %H:%M")
            time_diff = reminder_time - current_time
            
            # Notify 1 minute before
            if time_diff.total_seconds() > 0 and time_diff.total_seconds() <= 60:
                notification.notify(
                    title=f"Reminder Coming Up: {reminder['title']}",
                    message="This reminder is due in 1 minute!",
                    app_icon=None,
                    timeout=10,
                )
                text_to_speech(f"Attention! Your reminder for {reminder['title']} is due in 1 minute")
            
            # When reminder is due
            elif current_time >= reminder_time:
                notification.notify(
                    title=f"Reminder: {reminder['title']}",
                    message=reminder["description"] if reminder["description"] else "Task time!",
                    app_icon=None,
                    timeout=10,
                )
                text_to_speech(f"Task time! Your reminder for {reminder['title']} is now. {reminder['description'] if reminder['description'] else ''}")
                reminder["completed"] = True
                save_reminders()

def check_timers():
    """Check for completed timers and notify user."""
    current_time = datetime.datetime.now()
    for timer in active_timers[:]:  # Create a copy of the list to iterate
        time_remaining = (timer["end_time"] - current_time).total_seconds()
        
        # Notify when 1 minute remaining
        if 59 <= time_remaining <= 61:
            notification.notify(
                title=f"Timer: {timer['title']}",
                message="1 minute remaining!",
                app_icon=None,
                timeout=10,
            )
            text_to_speech(f"1 minute remaining in your {timer['title']} timer")
        
        # Timer complete
        elif current_time >= timer["end_time"]:
            notification.notify(
                title=f"Timer Complete: {timer['title']}",
                message=f"Your {timer['duration']} minute timer is complete! Task time!",
                app_icon=None,
                timeout=10,
            )
            text_to_speech(f"Time's up! Your {timer['duration']} minute timer is complete. Task time!")
            active_timers.remove(timer)

def extract_reminder_info(command):
    """Extract reminder information from natural language command."""
    command = command.lower()
    
    # Common patterns for reminder commands
    reminder_patterns = [
        r"remind me to (.*?) (?:on|at|by) (.*)",
        r"set (?:a )?reminder (?:to|for) (.*?) (?:on|at|by) (.*)",
        r"add (?:a )?reminder (?:to|for) (.*?) (?:on|at|by) (.*)",
        r"remind me about (.*?) (?:on|at|by) (.*)",
    ]
    
    # Try each pattern
    for pattern in reminder_patterns:
        match = re.search(pattern, command)
        if match:
            title = match.group(1).strip()
            datetime_str = match.group(2).strip()
            return title, datetime_str
            
    # If no pattern matches but contains keywords, try to extract information
    keywords = ["remind", "reminder", "schedule", "task"]
    if any(keyword in command for keyword in keywords):
        # Split the command into words
        words = command.split()
        
        # Try to find date/time information
        date_time_str = " ".join(words[-3:])  # Take last 3 words as potential date/time
        title = " ".join(words[2:-3])  # Rest of the words as title
        
        return title.strip(), date_time_str.strip()
    
    return None, None

def process_natural_language_command(command):
    """Process natural language commands for reminders."""
    try:
        title, datetime_str = extract_reminder_info(command)
        
        if title and datetime_str:
            reminder_time = parse_datetime_string(datetime_str)
            
            # Add the reminder
            reminder = {
                "title": title,
                "time": reminder_time.strftime("%Y-%m-%d %H:%M"),
                "description": "",  # Can be updated later if needed
                "completed": False
            }
            reminders.append(reminder)
            save_reminders()
            
            response = f"Okay, I'll remind you about {title} on {reminder_time.strftime('%B %d at %I:%M %p')}"
            text_to_speech(response)
            return True
            
        return False
    except Exception as e:
        print(f"Error processing command: {str(e)}")
        return False

def delete_reminder(title=None, index=None):
    """Delete a reminder by title or index."""
    try:
        if title:
            # Find reminders with matching title (case-insensitive)
            matching_reminders = [r for r in reminders if r["title"].lower() == title.lower()]
            if matching_reminders:
                for reminder in matching_reminders:
                    reminders.remove(reminder)
                save_reminders()
                text_to_speech(f"Deleted reminder for {title}")
                return True
            else:
                text_to_speech(f"No reminder found with title {title}")
                return False
        elif index is not None:
            if 0 <= index < len(reminders):
                deleted_reminder = reminders.pop(index)
                save_reminders()
                text_to_speech(f"Deleted reminder for {deleted_reminder['title']}")
                return True
            else:
                text_to_speech("Invalid reminder number")
                return False
        return False
    except Exception as e:
        print(f"Error deleting reminder: {str(e)}")
        return False

def list_reminders():
    """List all active reminders."""
    active_reminders = [r for r in reminders if not r["completed"]]
    if active_reminders:
        text_to_speech("Here are your active reminders:")
        for i, reminder in enumerate(active_reminders, 1):
            reminder_time = datetime.datetime.strptime(reminder["time"], "%Y-%m-%d %H:%M")
            text_to_speech(f"Number {i}: {reminder['title']} on {reminder_time.strftime('%B %d at %I:%M %p')}")
        return True
    else:
        text_to_speech("You have no active reminders")
        return False

def list_timers():
    """List all active timers."""
    if active_timers:
        text_to_speech("Here are your active timers:")
        current_time = datetime.datetime.now()
        for i, timer in enumerate(active_timers, 1):
            remaining_time = (timer["end_time"] - current_time).total_seconds() / 60
            text_to_speech(f"Timer {i}: {timer['title']}, {int(remaining_time)} minutes remaining")
        return True
    else:
        text_to_speech("You have no active timers")
        return False

def delete_timer(index=None, title=None):
    """Delete a timer by index or title."""
    try:
        if title:
            # Find timers with matching title (case-insensitive)
            matching_timers = [t for t in active_timers if t["title"].lower() == title.lower()]
            if matching_timers:
                for timer in matching_timers:
                    active_timers.remove(timer)
                text_to_speech(f"Deleted timer for {title}")
                return True
            else:
                text_to_speech(f"No timer found with title {title}")
                return False
        elif index is not None:
            if 0 <= index < len(active_timers):
                deleted_timer = active_timers.pop(index)
                text_to_speech(f"Deleted timer for {deleted_timer['title']}")
                return True
            else:
                text_to_speech("Invalid timer number")
                return False
        return False
    except Exception as e:
        print(f"Error deleting timer: {str(e)}")
        return False

def get_current_location():
    """Get the current location using IP-based geolocation."""
    try:
        # Using ip-api.com (free, no API key needed)
        location = geocoder.ip('me')
        if location.ok:
            return {
                'latitude': location.lat,
                'longitude': location.lng,
                'address': location.address,
                'city': location.city,
                'state': location.state,
                'country': location.country
            }
        else:
            print("Error getting location:", location.error)
            return None
    except Exception as e:
        print(f"Error in get_current_location: {str(e)}")
        return None

def create_location_map(latitude, longitude, save_path=None):
    """Create an interactive map with the current location."""
    try:
        # Create a map centered at the current location
        location_map = folium.Map(location=[latitude, longitude], zoom_start=15)
        
        # Add a marker for the current location
        folium.Marker(
            [latitude, longitude],
            popup='Current Location',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(location_map)
        
        # Save the map to an HTML file
        if save_path is None:
            save_path = os.path.join(os.path.expanduser('~'), 'robot_location_map.html')
        
        location_map.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error creating map: {str(e)}")
        return None

def show_current_location():
    """Get and display the current location."""
    try:
        location_info = get_current_location()
        if location_info:
            # Announce the location
            location_text = f"Current location: {location_info['address']}. "
            location_text += f"Coordinates: {location_info['latitude']:.6f}, {location_info['longitude']:.6f}"
            text_to_speech(location_text)
            
            # Create and show the map
            map_path = create_location_map(location_info['latitude'], location_info['longitude'])
            if map_path:
                webbrowser.open('file://' + os.path.abspath(map_path))
                return True
        return False
    except Exception as e:
        print(f"Error showing location: {str(e)}")
        text_to_speech("Sorry, I couldn't get the current location.")
        return False

def extract_numbers(text):
    """Extract numbers from text, including word form."""
    try:
        # Replace written numbers with digits
        words = text.lower().split()
        numbers = []
        i = 0
        while i < len(words):
            try:
                # Try to convert word or phrase to number
                num_words = []
                while i < len(words) and (words[i].replace('.', '').replace('-', '').isdigit() or 
                      words[i] in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                                 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                                 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
                                 'hundred', 'thousand', 'million', 'billion']):
                    num_words.append(words[i])
                    i += 1
                if num_words:
                    try:
                        # Try to convert word to number
                        number = w2n.word_to_num(' '.join(num_words))
                        numbers.append(number)
                    except ValueError:
                        # If word_to_num fails, try parsing as regular number
                        try:
                            number = float(''.join(num_words))
                            numbers.append(number)
                        except ValueError:
                            pass
            except ValueError:
                i += 1
            if not num_words:
                i += 1
        return numbers
    except Exception as e:
        print(f"Error extracting numbers: {str(e)}")
        return []

def perform_math_operation(operation, numbers):
    """Perform mathematical operation on the given numbers."""
    try:
        if len(numbers) < 2:
            return None, "I need at least two numbers to perform calculations."
        
        if "add" in operation or "plus" in operation or "sum" in operation:
            result = sum(numbers)
            operation_text = "sum"
        elif "subtract" in operation or "minus" in operation or "difference" in operation:
            result = numbers[0] - sum(numbers[1:])
            operation_text = "difference"
        elif "multiply" in operation or "times" in operation or "product" in operation:
            result = math.prod(numbers)
            operation_text = "product"
        elif "divide" in operation or "division" in operation:
            if 0 in numbers[1:]:
                return None, "Cannot divide by zero."
            result = numbers[0]
            for num in numbers[1:]:
                result /= num
            operation_text = "division"
        elif "average" in operation or "mean" in operation:
            result = sum(numbers) / len(numbers)
            operation_text = "average"
        elif "power" in operation or "exponent" in operation:
            if len(numbers) != 2:
                return None, "Power operation needs exactly two numbers."
            result = math.pow(numbers[0], numbers[1])
            operation_text = "power"
        elif "square root" in operation or "sqrt" in operation:
            if len(numbers) != 1:
                return None, "Square root operation needs exactly one number."
            if numbers[0] < 0:
                return None, "Cannot calculate square root of a negative number."
            result = math.sqrt(numbers[0])
            operation_text = "square root"
        else:
            return None, "I don't understand that mathematical operation."
        
        # Round to 6 decimal places if it's a float
        if isinstance(result, float):
            result = round(result, 6)
        
        return result, operation_text
    except Exception as e:
        print(f"Error in calculation: {str(e)}")
        return None, "Sorry, I couldn't perform that calculation."

def process_math_command(command):
    """Process mathematical commands and return result."""
    try:
        # Extract numbers from the command
        numbers = extract_numbers(command)
        if not numbers:
            text_to_speech("I couldn't find any numbers in your command.")
            return False
        
        # Perform the operation
        result, operation_text = perform_math_operation(command, numbers)
        if result is not None:
            # Format numbers for speech
            numbers_text = ", ".join(str(num) for num in numbers[:-1])
            if len(numbers) > 1:
                numbers_text += f" and {numbers[-1]}"
            
            # Announce result
            text_to_speech(f"The {operation_text} of {numbers_text} is {result}")
            return True
        else:
            text_to_speech(operation_text)  # operation_text contains error message
            return False
    except Exception as e:
        print(f"Error processing math command: {str(e)}")
        text_to_speech("Sorry, I couldn't process that mathematical operation.")
        return False

def fetch_news(category='general', num_articles=5):
    """Fetch news from Google News RSS feeds."""
    try:
        if category.lower() not in NEWS_CATEGORIES:
            category = 'general'
        
        url = NEWS_CATEGORIES[category.lower()]
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features='xml')
        
        articles = []
        items = soup.findAll('item')
        
        for i, item in enumerate(items[:num_articles]):
            article = {
                'title': item.title.text,
                'link': item.link.text,
                'published': item.pubDate.text if item.pubDate else None,
                'description': item.description.text if item.description else None
            }
            articles.append(article)
        
        return articles
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return None

def read_news_headlines(category='general', num_articles=5):
    """Read out news headlines for the specified category."""
    try:
        articles = fetch_news(category, num_articles)
        if articles:
            text_to_speech(f"Here are the top {num_articles} {category} news headlines:")
            for i, article in enumerate(articles, 1):
                # Clean up the title by removing HTML tags and special characters
                title = BeautifulSoup(article['title'], 'html.parser').get_text()
                text_to_speech(f"News {i}: {title}")
                
                # Brief pause between headlines
                time.sleep(0.5)
            return True
        else:
            text_to_speech(f"Sorry, I couldn't fetch {category} news at the moment.")
            return False
    except Exception as e:
        print(f"Error reading news: {str(e)}")
        text_to_speech("Sorry, I encountered an error while fetching the news.")
        return False

def save_article_to_knowledge_base(article):
    """Save a news article to the knowledge base."""
    try:
        title = BeautifulSoup(article['title'], 'html.parser').get_text()
        key = f"article_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        value = {
            'title': title,
            'link': article['link'],
            'published': article['published'],
            'description': article['description']
        }
        add_to_knowledge_base('news', key, value)
        return True
    except Exception as e:
        print(f"Error saving article: {str(e)}")
        return False

def save_weather_to_knowledge_base(location, weather_data):
    """Save weather data to knowledge base."""
    try:
        key = f"weather_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        value = {
            'location': location,
            'data': weather_data,
            'timestamp': datetime.datetime.now().isoformat()
        }
        add_to_knowledge_base('weather', key, value)
        return True
    except Exception as e:
        print(f"Error saving weather data: {str(e)}")
        return False

def get_weather(location=None):
    """Get weather information for a location using web scraping."""
    try:
        # If no location provided, use IP-based location
        if not location:
            loc = geocoder.ip('me')
            if loc.ok:
                location = f"{loc.city}, {loc.state}, {loc.country}"
            else:
                return None, "Couldn't determine your location"

        # Encode location for URL
        encoded_location = urllib.parse.quote(location)
        
        # Weather.com search URL
        search_url = f"https://weather.com/weather/today/l/{encoded_location}"
        
        # Send request with headers to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            return None, "Couldn't fetch weather data"
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract weather information
        weather_data = {}
        
        # Current temperature
        temp_elem = soup.find('span', {'data-testid': 'TemperatureValue'})
        if temp_elem:
            weather_data['temperature'] = temp_elem.text.strip()
        
        # Weather condition
        condition_elem = soup.find('div', {'data-testid': 'wxPhrase'})
        if condition_elem:
            weather_data['condition'] = condition_elem.text.strip()
        
        # Humidity
        humidity_elem = soup.find('span', {'data-testid': 'PercentageValue'})
        if humidity_elem:
            weather_data['humidity'] = humidity_elem.text.strip()
        
        # Wind
        wind_elem = soup.find('span', {'data-testid': 'Wind'})
        if wind_elem:
            weather_data['wind'] = wind_elem.text.strip()
        
        if weather_data:
            return weather_data, None
        else:
            return None, "Couldn't extract weather information"
            
    except Exception as e:
        print(f"Error getting weather: {str(e)}")
        return None, str(e)

def announce_weather(location=None):
    """Announce weather information using text-to-speech."""
    try:
        weather_data, error = get_weather(location)
        
        if error:
            text_to_speech(f"Sorry, {error}")
            return False
            
        if weather_data:
            # Construct weather announcement
            announcement = f"Here's the weather"
            if location:
                announcement += f" for {location}"
            announcement += ": "
            
            if 'temperature' in weather_data:
                announcement += f"Temperature is {weather_data['temperature']}, "
            
            if 'condition' in weather_data:
                announcement += f"conditions are {weather_data['condition']}, "
            
            if 'humidity' in weather_data:
                announcement += f"humidity is {weather_data['humidity']}, "
            
            if 'wind' in weather_data:
                announcement += f"wind is {weather_data['wind']}"
            
            text_to_speech(announcement)
            
            # Save to knowledge base
            save_weather_to_knowledge_base(location, weather_data)
            return True
        else:
            text_to_speech("Sorry, I couldn't get the weather information")
            return False
            
    except Exception as e:
        print(f"Error announcing weather: {str(e)}")
        text_to_speech("Sorry, I encountered an error while getting the weather")
        return False

def process_voice_commands(command_queue):
    """Process voice commands from the queue."""
    while True:
        try:
            # Get command from the queue instead of directly listening
            # This allows the non_blocking_listen function to handle interruptions
            if not command_queue.empty():
                cmd_type, command = command_queue.get()
                if cmd_type != 'VOICE_INPUT' or not command:
                    continue
                
                # First check if there's a response in the knowledge base
                kb_response = get_knowledge_base_response(command)
                if kb_response:
                    text_to_speech(kb_response)
                    continue
                
                # Weather commands
                if "weather" in command:
                    # Try to extract location if provided
                    location = None
                    if "in" in command:
                        location = command.split("in")[-1].strip()
                    elif "for" in command:
                        location = command.split("for")[-1].strip()
                    announce_weather(location)
                    continue
                
                # News commands
                elif "news" in command:
                    category = 'general'
                    num_articles = 5
                    
                    # Extract category if specified
                    for cat in NEWS_CATEGORIES.keys():
                        if cat in command:
                            category = cat
                            break
                    
                    # Extract number of articles if specified
                    numbers = extract_numbers(command)
                    if numbers and 1 <= numbers[0] <= 10:
                        num_articles = int(numbers[0])
                    
                    read_news_headlines(category, num_articles)
                    continue
                
                # Math operations
                math_keywords = ["add", "plus", "sum", "subtract", "minus", "difference", 
                               "multiply", "times", "product", "divide", "division",
                               "average", "mean", "power", "exponent", "square root", "sqrt"]
                
                if any(keyword in command for keyword in math_keywords):
                    process_math_command(command)
                    continue
                
                # Location commands
                elif "location" in command or "where am i" in command or "show map" in command:
                    show_current_location()
                    continue
                
                # Wikipedia search
                elif "what is" in command or "who is" in command or "tell me about" in command:
                    query = command
                    for phrase in ["what is", "who is", "tell me about"]:
                        query = query.replace(phrase, "")
                    query = query.strip()
                    
                    if query:
                        result, success = search_wikipedia(query)
                        if success:
                            text_to_speech(result)
                            # Save to knowledge base
                            add_to_knowledge_base('facts', query, result)
                        else:
                            text_to_speech(f"I'm not sure about {query}. Would you like me to search for something else?")
                    continue
                
                # Try to process as a natural language reminder command first
                elif any(keyword in command for keyword in ["remind", "reminder", "schedule", "task"]):
                    # Check for delete commands
                    if "delete" in command or "remove" in command:
                        if "reminder" in command:
                            # Try to extract reminder title or number
                            if "number" in command:
                                try:
                                    index = int(''.join(filter(str.isdigit, command))) - 1
                                    delete_reminder(index=index)
                                except ValueError:
                                    text_to_speech("Please specify a valid reminder number")
                            else:
                                # Extract title after "delete/remove reminder"
                                title = command.split("reminder")[-1].strip()
                                delete_reminder(title=title)
                        continue
                    # Check for list commands
                    elif "list" in command or "show" in command:
                        if "reminder" in command:
                            list_reminders()
                        elif "timer" in command:
                            list_timers()
                        continue
                    # Process natural language reminder command
                    elif process_natural_language_command(command):
                        continue

                # Timer commands
                elif "timer" in command:
                    if "delete" in command or "remove" in command:
                        if "number" in command:
                            try:
                                index = int(''.join(filter(str.isdigit, command))) - 1
                                delete_timer(index=index)
                            except ValueError:
                                text_to_speech("Please specify a valid timer number")
                        else:
                            # Extract title after "delete/remove timer"
                            title = command.split("timer")[-1].strip()
                            delete_timer(title=title)
                    elif "list" in command or "show" in command:
                        list_timers()
                    elif "set" in command or "start" in command:
                        try:
                            duration = int(''.join(filter(str.isdigit, command)))
                            if duration > 0:
                                start_timer(duration)
                            else:
                                text_to_speech("Please specify a valid duration in minutes")
                        except:
                            text_to_speech("Please specify how many minutes for the timer")
                
                # Knowledge base commands
                elif "what do you know" in command or "show knowledge" in command:
                    if knowledge_base:
                        text_to_speech(f"I know about {len(knowledge_base)} topics. Here are some examples: " + 
                                     ", ".join(list(knowledge_base.keys())[:3]))
                    else:
                        text_to_speech("My knowledge base is currently empty")
                
                # Date and time commands
                elif any(phrase in command for phrase in ["current date", "today's date", "what date", "what is the date", "what day is it", "what is today"]):
                    current_date = datetime.datetime.now()
                    date_str = current_date.strftime("%A, %B %d, %Y")
                    text_to_speech(f"Today's date is {date_str}")
                    continue
                
                # Time commands
                elif any(phrase in command for phrase in ["current time", "what time", "what is the time", "tell me the time"]):
                    current_time = datetime.datetime.now()
                    time_str = current_time.strftime("%I:%M %p")
                    text_to_speech(f"The current time is {time_str}")
                    continue
                
                # Other existing commands
                elif "quit" in command or "exit" in command:
                    command_queue.put(('QUIT', None))
                    break
                else:
                    command_queue.put(('VOICE_INPUT', command))

        except Exception as e:
            print(f"Error in voice command processing: {str(e)}")
            continue

NEWS_CATEGORIES = {
    'general': 'https://news.google.com/rss',
    'technology': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB',
    'business': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB',
    'science': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB',
    'health': 'https://news.google.com/rss/topics/CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ',
    'sports': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB',
    'entertainment': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB'
}

def get_knowledge_base_response(query):
    """Get a response from the knowledge base based on a user query.
    
    Args:
        query: The user's query string
    
    Returns:
        A response string if found, None otherwise
    """
    try:
        query = query.lower().strip()
        
        # Check for direct command matches
        command_response = get_from_knowledge_base('commands', query)
        if command_response:
            return command_response
            
        # Handle date and time queries in knowledge base
        if any(phrase in query for phrase in ["what time is it", "current time", "tell me the time"]):
            current_time = datetime.datetime.now()
            return f"The current time is {current_time.strftime('%I:%M %p')}"
            
        if any(phrase in query for phrase in ["what date is it", "current date", "today's date", "what day is it", "what is today"]):
            current_date = datetime.datetime.now()
            return f"Today's date is {current_date.strftime('%A, %B %d, %Y')}"
            
        # Check for fact queries
        for fact_key in get_from_knowledge_base('facts', None) or {}:
            if fact_key.lower() in query:
                return get_from_knowledge_base('facts', fact_key)
        
        # Check for weather history
        if 'weather' in query or 'temperature' in query:
            weather_data = get_from_knowledge_base('weather', None)
            if weather_data and len(weather_data) > 0:
                # Get the most recent weather entry
                latest_key = sorted(weather_data.keys())[-1]
                weather_entry = weather_data[latest_key]
                location = weather_entry.get('location', 'unknown location')
                data = weather_entry.get('data', {})
                if data:
                    return f"Last recorded weather for {location}: Temperature {data.get('temperature', 'unknown')}, Conditions {data.get('condition', 'unknown')}"
        
        # Check for news queries
        if 'news' in query or 'headlines' in query:
            news_data = get_from_knowledge_base('news', None)
            if news_data and len(news_data) > 0:
                # Get the most recent news entry
                latest_key = sorted(news_data.keys())[-1]
                news_entry = news_data[latest_key]
                return f"Latest news: {news_entry.get('title', 'No title available')}"
        
        # Get general information about the robot
        if 'your name' in query or 'who are you' in query:
            robot_name = get_from_knowledge_base('general', 'robot_name')
            if robot_name:
                return f"I am {robot_name}, your helpful robot companion."
        
        # No matching information found
        return None
        
    except Exception as e:
        print(f"Error retrieving from knowledge base: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech."""
    global speech_in_progress, stop_speech
    
    try:
        # Set speech in progress flag
        with speech_lock:
            speech_in_progress = True
            stop_speech = False
        
        # Create a callback to check for stop command during speech
        def on_word(name, location, length):
            global stop_speech
            if stop_speech:
                engine.stop()
                return
        
        # Connect the callback if possible
        try:
            engine.connect('started-word', on_word)
        except Exception as e:
            print(f"Warning: Could not connect word callback: {str(e)}")
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        # Reset speech flags
        with speech_lock:
            speech_in_progress = False
            stop_speech = False
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        # Make sure to reset flags even on error
        with speech_lock:
            speech_in_progress = False
            stop_speech = False

def main():
    """Main function to run the vision and voice system."""
    global last_voice_command, continuous_listening_enabled
    
    # Load known faces and knowledge base
    load_known_faces()
    load_knowledge_base()
    load_reminders()  # Load saved reminders
    
    # Initialize components
    command_queue = Queue()
    
    # Start the continuous listening thread for interruptions and priority questions
    continuous_thread = threading.Thread(target=continuous_listen, args=(command_queue,))
    continuous_thread.daemon = True
    continuous_thread.start()
    
    # Start the command processing thread
    voice_thread = threading.Thread(target=process_voice_commands, args=(command_queue,))
    voice_thread.daemon = True
    voice_thread.start()
    
    # Timer for checking reminders and timers
    last_check_time = time.time()
    check_interval = 30  # Check every 30 seconds
    
    try:
        # Initialize camera with improved error handling
        cap = init_camera()
    except Exception as e:
        print(f"Fatal error: Could not initialize camera: {str(e)}")
        return

    # Create window with proper backend
    window_name = create_window('Vision and Voice System')
    if window_name is None:
        print("Error: Could not create display window")
        return

    frame_count = 0
    start_time = time.time()
    last_voice_command = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame_count += 1

            # Face detection and recognition
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4, minSize=(30, 30))
            
            if len(faces) > 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (x, y, w, h), face_encoding in zip(faces, face_encodings):
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Face recognition
                    name = "Unknown"
                    if known_face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        if True in matches:
                            name = known_face_names[matches.index(True)]
                    
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Emotion detection (every 5th frame)
                    if frame_count % 5 == 0:
                        face_roi = frame[y:y+h, x:x+w]
                        emotions = emotion_detector.detect_emotions(face_roi)
                        if emotions:
                            dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])[0]
                            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y + h + 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Hand gesture detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture = recognize_gesture(hand_landmarks.landmark)
                    if gesture:
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Object detection (every 3rd frame)
            if frame_count % 3 == 0:
                results = model_yolo.predict(source=frame, conf=0.5, iou=0.45)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = model_yolo.names[cls]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Process voice commands
            try:
                while not command_queue.empty():
                    cmd_type, cmd_data = command_queue.get_nowait()
                    if cmd_type == 'QUIT':
                        return
                    elif cmd_type == 'VOICE_INPUT':
                        last_voice_command = cmd_data
            except Exception as e:
                print(f"Error processing voice command: {str(e)}")

            # Display last voice command
            if last_voice_command:
                cv2.putText(frame, f"Voice: {last_voice_command}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check reminders and timers periodically
            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                check_reminders()
                check_timers()
                last_check_time = current_time

            # Display frame with error handling
            try:
                cv2.imshow(window_name, frame)
            except Exception as e:
                print(f"Error displaying frame: {str(e)}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping the program...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        # Shutdown continuous listening
        continuous_listening_enabled = False
        voice_thread.join(timeout=1)
        continuous_thread.join(timeout=1)

if __name__ == "__main__":
    main()
