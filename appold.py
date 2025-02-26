#app.py
from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import subprocess
import random
import imaplib
import time
import email
import atexit
import string
import logging
from email.header import decode_header
from vosk import Model, KaldiRecognizer
import json
import pyaudio
import wave
import pyttsx3
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
EMAIL = "sreeragrok1@gmail.com"
PASSWORD = "lzpe wloy mzpc mzph"
viewed_email_ids = set()

# Directory to save the generated audio
OUTPUT_DIR = "./output/result/LJSpeech"

# Required directories
required_dirs = [
    OUTPUT_DIR,
    'config/LJSpeech',
]

# Create required directories
for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

def check_required_files():
    required_paths = {
        'python_executable': r'D:/Main Project/FastSpeech2/envfin/Scripts/python.exe',
        'synthesize_script': 'synthesize.py',
        'config_preprocess': 'config/LJSpeech/preprocess.yaml',
        'config_model': 'config/LJSpeech/model.yaml',
        'config_train': 'config/LJSpeech/train.yaml'
    }

    missing_files = []
    for name, path in required_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    return missing_files

# Initialize Vosk model
if not os.path.exists("D:/Main Project/FastSpeech2/vosk-model-small-en-us-0.15"):
    print("Please download the model from https://alphacephei.com/vosk/models")
    exit(1)

model = Model("D:/Main Project/FastSpeech2/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Initialize TTS engine
engine = pyttsx3.init()

# Global variables for continuous listening
listening = True
command_queue = queue.Queue()

def text_to_speech(text, output_file):
    try:
        engine.save_to_file(text, output_file)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")

def continuous_listening():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=16000,
                   input=True,
                   frames_per_buffer=8000)
    
    logger.info("Continuous listening started...")
    
    while listening:
        try:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if result["text"]:
                    logger.info(f"Recognized: {result['text']}")
                    command_queue.put(result["text"].lower())
                    process_command(result["text"].lower())
        except Exception as e:
            logger.error(f"Error in continuous listening: {e}")
            continue

    stream.stop_stream()
    stream.close()
    p.terminate()

def process_command(command):
    if "read email" in command:
        logger.info("Processing 'read email' command...")
        emails = fetch_unread_emails()
        if not emails:
            response_text = "No new emails found."
        else:
            response_text = "Reading your emails:\n"
            for i, email_data in enumerate(emails, 1):
                sender = email_data['from'].split('<')[0].strip() if '<' in email_data['from'] else email_data['from']
                response_text += f"Email {i}: From {sender}. Subject: {email_data['subject']}. Message: {email_data['body']}\n"
        
        # Generate a unique file ID
        file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_file = f"{file_id}.wav"
        output_file_path = os.path.join(OUTPUT_DIR, output_file)

        try:
            # Use pyttsx3 for speech synthesis
            text_to_speech(response_text, output_file_path)
            
            # Verify file exists and has content
            if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
                # Store the filename in app config for the template to access
                app.config['CURRENT_AUDIO'] = output_file
                logger.info(f"Audio file created successfully: {output_file}")
                return True
            else:
                logger.error("Audio file was not created or is empty")
                return False
                
        except Exception as e:
            logger.error(f"Error in email synthesis: {e}")
            return False
            
def start_listening_thread():
    thread = threading.Thread(target=continuous_listening)
    thread.daemon = True
    thread.start()

def fetch_unread_emails(mark_as_seen=False):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox", readonly=not mark_as_seen)
        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()
        emails = []

        for email_id in reversed(email_ids):
            status, msg_data = mail.fetch(email_id, "(RFC822)")

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])

                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")

                    from_ = msg.get("From")

                    body = None
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))

                            if content_type in ["text/plain", "text/html"]:
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body = payload.decode(errors='ignore')
                                    if content_type == "text/plain":
                                        break
                    else:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors='ignore')

                    emails.append({
                        "subject": subject,
                        "from": from_,
                        "body": body.strip() if body else "No body content"
                    })
                    
                    viewed_email_ids.add(email_id)

        mail.logout()
        return emails
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        return []

@app.route('/')
def home():
    emails = fetch_unread_emails()
    audio_file = app.config.get('CURRENT_AUDIO')
    return render_template('index.html', emails=emails, audio_file=audio_file)

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype='audio/wav')

@app.route('/check_commands')
def check_commands():
    try:
        command = command_queue.get_nowait()
        return jsonify({
            'command': command,
            'reload': 'read email' in command.lower()
        })
    except queue.Empty:
        return jsonify({'command': None, 'reload': False})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        emails = fetch_unread_emails()
        if not emails:
            return "No emails to synthesize"

        combined_text = "\n\n".join([
            f"From: {email['from'].split('<')[0].strip() if '<' in email['from'] else email['from']}. "
            f"Subject: {email['subject']}. "
            f"Body: {email['body']}."
            for email in emails
        ])   
        
        file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_file = f"{file_id}.wav"
        output_file_path = os.path.join(OUTPUT_DIR, output_file)

        # Verify python path exists
        python_bin = r'D:/Main Project/FastSpeech2/envfin/Scripts/python.exe'
        if not os.path.exists(python_bin):
            return f"Error: Python executable not found at {python_bin}"

        # Verify synthesize.py exists
        synthesize_script = os.path.join(os.path.dirname(__file__), 'synthesize.py')
        if not os.path.exists(synthesize_script):
            return "Error: synthesize.py not found"

        command = [
            python_bin,
            synthesize_script,
            '--text', combined_text,
            '--restore_step', '900000',
            '--mode', 'single',
            '--file_id', file_id,
            '-p', 'config/LJSpeech/preprocess.yaml',
            '-m', 'config/LJSpeech/model.yaml',
            '-t', 'config/LJSpeech/train.yaml'
        ]
        
        logger.debug(f"Command: {' '.join(command)}")
        logger.debug(f"Working directory: {os.getcwd()}")

        # Run the command with full error capture
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Synthesis failed: {stderr}")
            return f"Error: {stderr}"

        if os.path.exists(output_file_path):
            return render_template('index.html', audio_file=output_file)
        else:
            return "Error: Audio file was not generated"

    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/synthesize_email', methods=['POST'])
def synthesize_email():
    try:
        email_subject = request.form['email_subject']
        email_from = request.form['email_from']
        email_body = request.form['email_body']
        
        if '<' in email_from:
            email_from = email_from.split('<')[0].strip()

        email_text = f"From: {email_from}. Subject: {email_subject}. Body: {email_body}."

        file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_file = f"{file_id}.wav"
        output_file_path = os.path.join(OUTPUT_DIR, output_file)

        python_bin = r'D:/Main Project/FastSpeech2/envfin/Scripts/python.exe'
        if not os.path.exists(python_bin):
            return f"Error: Python executable not found at {python_bin}"

        synthesize_script = os.path.join(os.path.dirname(__file__), 'synthesize.py')
        if not os.path.exists(synthesize_script):
            return "Error: synthesize.py not found"

        command = [
            python_bin,
            synthesize_script,
            '--text', email_text,
            '--restore_step', '900000',
            '--mode', 'single',
            '--file_id', file_id,
            '-p', 'config/LJSpeech/preprocess.yaml',
            '-m', 'config/LJSpeech/model.yaml',
            '-t', 'config/LJSpeech/train.yaml'
        ]

        logger.debug(f"Command: {' '.join(command)}")
        logger.debug(f"Working directory: {os.getcwd()}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Single email synthesis failed: {stderr}")
            return f"Error: {stderr}"

        if os.path.exists(output_file_path):
            return render_template('index.html', audio_file=output_file)
        else:
            return "Error: Audio file was not generated"

    except Exception as e:
        logger.error(f"Single email synthesis error: {str(e)}")
        return f"Error: {str(e)}"

def mark_as_read_on_exit():
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        
        for email_id in viewed_email_ids:
            mail.store(email_id, '+FLAGS', '\\Seen')
        
        mail.logout()
    except Exception as e:
        logger.error(f"Error marking emails as read on exit: {e}")

def cleanup():
    global listening
    listening = False
    if engine:
        try:
            engine.stop()
        except:
            pass
    mark_as_read_on_exit()

# Register cleanup functions
atexit.register(cleanup)

if __name__ == '__main__':
    # Check for missing files
    missing_files = check_required_files()
    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"- {file}")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize TTS engine properties
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    
    # Start the continuous listening thread
    start_listening_thread()
    
    # Run the Flask app
    app.run(debug=True, use_reloader=False)