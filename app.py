# integrated_app.py
from flask import Flask, request, render_template, send_from_directory, jsonify, Response, session, redirect, url_for
import os
import subprocess
import random
import imaplib
import time
import email
import atexit
import string
import logging
import cv2
import numpy as np
import dlib
import pickle
from email.header import decode_header
from vosk import Model, KaldiRecognizer
import json
import pyaudio
import wave
import pyttsx3
import threading
import queue
from mtcnn import MTCNN
from threading import Thread

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change this to a random secret key

# Email configuration
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

# -------------------------
# Face Recognition Functionality
# -------------------------

# Path to the saved face encodings
ENCODINGS_FILE = "face_encodings.pkl"

# Initialize MTCNN detector
detector = MTCNN()

# Load dlib models
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
except Exception as e:
    logger.error(f"Error loading dlib models: {e}")
    logger.warning("Face recognition functionality may not work properly!")

# Load known face encodings
face_encoding_dict = {}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        face_encoding_dict = pickle.load(f)
    logger.info(f"Loaded {len(face_encoding_dict)} face encodings")
else:
    logger.warning("No face encodings found. Please run facereg.py first to generate encodings.")

# Global variables for face verification
is_verifying = False
verification_result = None
verification_start_time = None
recognized_name = None

def get_face_encoding(image, face):
    """Get face encoding from image and face coordinates"""
    x, y, w, h = face
    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(image, dlib_rect)
    return np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1))

def find_best_match(unknown_encoding, threshold=0.5):
    """Find best match for a face encoding"""
    if not face_encoding_dict:
        return "Intruder"
    
    known_names = list(face_encoding_dict.keys())
    known_encodings = np.vstack(list(face_encoding_dict.values()))
    
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    min_idx = np.argmin(distances)
    
    return "Intruder" if distances[min_idx] > threshold else known_names[min_idx]

def verify_face():
    """Face verification process that runs in a separate thread"""
    global is_verifying, verification_result, verification_start_time, recognized_name
    
    verification_start_time = time.time()
    consecutive_matches = 0
    last_match = None
    
    cap = cv2.VideoCapture(0)
    
    while is_verifying and time.time() - verification_start_time < 10:  # 10 sec timeout
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_faces(rgb_frame)
        
        if detected_faces:
            best_face = max(detected_faces, key=lambda f: f['box'][2] * f['box'][3])
            box = best_face['box']
            
            try:
                encoding = get_face_encoding(rgb_frame, box)
                name = find_best_match(encoding)
                
                if name != "Intruder":
                    if last_match == name:
                        consecutive_matches += 1
                    else:
                        consecutive_matches = 1
                    
                    last_match = name
                    
                    # If we have 5 consecutive matches of the same person
                    if consecutive_matches >= 5:  # Roughly equivalent to 5 seconds
                        verification_result = True
                        recognized_name = name
                        break
                else:
                    consecutive_matches = 0
                    last_match = None
            except Exception as e:
                logger.error(f"Error during verification: {str(e)}")
                
    cap.release()
    
    # If verification wasn't successful but time is up
    if verification_result is None:
        verification_result = False
    
    is_verifying = False

def gen_frames():
    """Generate frames for video streaming"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # If we're verifying, add overlay text
        if is_verifying:
            elapsed = time.time() - verification_start_time
            remaining = max(0, 5 - elapsed)
            
            # Draw a black semi-transparent overlay at the bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            if verification_result is None:
                cv2.putText(frame, f"Verifying... {remaining:.1f}s", (10, frame.shape[0] - 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif verification_result:
                cv2.putText(frame, f"Verified as {recognized_name}! Redirecting...", (10, frame.shape[0] - 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Verification failed. Try again.", (10, frame.shape[0] - 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -------------------------
# Voice Recognition & Email Functionality
# -------------------------

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
try:
    if not os.path.exists("D:/Main Project/FastSpeech2/vosk-model-small-en-us-0.15"):
        logger.error("Voice model not found. Please download the model from https://alphacephei.com/vosk/models")
    else:
        model = Model("D:/Main Project/FastSpeech2/vosk-model-small-en-us-0.15")
        recognizer = KaldiRecognizer(model, 16000)
except Exception as e:
    logger.error(f"Error initializing voice model: {e}")

# Initialize TTS engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
except Exception as e:
    logger.error(f"Error initializing TTS engine: {e}")

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
    """Process voice commands for various actions"""
    logger.info(f"Processing command: {command}")
    
    # Email reading command
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
    
    # Login related commands - we just pass these through to the UI
    elif any(keyword in command for keyword in ["manual", "username", "password", "face", "submit", "log in", "login now"]):
        logger.info(f"Login command detected: {command}")
        # The command will be handled by the JavaScript in the login page
        return True
        
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
    if 'engine' in globals():
        try:
            engine.stop()
        except:
            pass
    mark_as_read_on_exit()

# Register cleanup functions
atexit.register(cleanup)

# -------------------------
# Login Routes
# -------------------------

@app.route('/login')
def login():
    """Login page route"""
    if 'user' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_verification')
def start_verification():
    """Start the face verification process"""
    global is_verifying, verification_result
    
    if not is_verifying:
        is_verifying = True
        verification_result = None
        verification_thread = Thread(target=verify_face)
        verification_thread.daemon = True
        verification_thread.start()
        
    return {'status': 'started'}

@app.route('/check_verification')
def check_verification():
    """Check the status of verification"""
    global is_verifying, verification_result, recognized_name
    
    if not is_verifying and verification_result:
        # Verification successful
        session['user'] = recognized_name
        return {'status': 'success', 'name': recognized_name}
    elif not is_verifying and verification_result is False:
        # Verification failed
        return {'status': 'failed'}
    else:
        # Still verifying
        return {'status': 'verifying'}

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# -------------------------
# Main Application Routes
# -------------------------

@app.route('/')
def home():
    """Main application home page"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    emails = fetch_unread_emails()
    audio_file = app.config.get('CURRENT_AUDIO')
    return render_template('index.html', emails=emails, audio_file=audio_file, username=session['user'])

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype='audio/wav')

@app.route('/check_commands')
def check_commands():
    try:
        # Non-blocking get from the queue with a timeout of 0.1 seconds
        command = command_queue.get(block=False)
        should_reload = "read email" in command  # Reload only if "read email" command detected
        return jsonify({
            'command': command,
            'reload': should_reload
        })
    except queue.Empty:
        return jsonify({'command': None, 'reload': False})

@app.route('/synthesize', methods=['POST'])
@app.route('/synthesize', methods=['POST'])
def synthesize():
    if 'user' not in session:
        return jsonify({'error': 'User not authenticated'}), 403

    try:
        emails = fetch_unread_emails()
        if not emails:
            return jsonify({'error': 'No emails to synthesize'})

        combined_text = "\n\n".join([
            f"From: {email['from']}. Subject: {email['subject']}. Body: {email['body']}."
            for email in emails
        ])   

        file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_file = f"{file_id}.wav"
        output_file_path = os.path.join(OUTPUT_DIR, output_file)

        # Perform text-to-speech conversion
        text_to_speech(combined_text, output_file_path)

        if os.path.exists(output_file_path):
            app.config['CURRENT_AUDIO'] = output_file
            return jsonify({'success': True, 'audio_file': output_file})
        else:
            return jsonify({'error': 'Audio file not generated'})

    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize_email', methods=['POST'])
def synthesize_email():
    if 'user' not in session:
        return redirect(url_for('login'))
        
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
            return render_template('index.html', audio_file=output_file, username=session['user'])
        else:
            return "Error: Audio file was not generated"

    except Exception as e:
        logger.error(f"Single email synthesis error: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/manual_login', methods=['POST'])
def manual_login():
    """Handle manual login with username/password"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Improved authentication logic
    # In a real application, you would use a database and secure password hashing
    valid_users = {
        "sree": "pass123",
        "admin": "admin123",
        "test": "test123"
    }
    
    if username in valid_users and valid_users[username] == password:
        session['user'] = username
        logger.info(f"Successful login for user: {username}")
        return redirect(url_for('home'))
    else:
        logger.warning(f"Failed login attempt for username: {username}")
        return render_template('login.html', error="Invalid username or password")

if __name__ == '__main__':
    # Check for missing files
    missing_files = check_required_files()
    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"- {file}")
        logger.warning("Some functionality may not work properly!")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start the continuous listening thread
    start_listening_thread()
    
    # Set default values
    app.config['CURRENT_AUDIO'] = None
    
    # Run the Flask app
    app.run(debug=True, use_reloader=False)