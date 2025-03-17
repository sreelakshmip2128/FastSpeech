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
import base64
import logging
import cv2
import numpy as np
import dlib
import pickle
from threading import Lock
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
# Add these imports at the top of integrated_app.py
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from email.utils import parsedate_to_datetime
user_credentials = {}
credentials_lock = Lock()
# Database configuration

# Set up Flask-Login

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

# Add these global variables
recognized_email = None
recognized_password = None
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
    """Find best match for a face encoding and return username and credentials"""
    if not face_encoding_dict:
        return "Intruder", None, None
    
    best_match = None
    best_distance = float('inf')
    
    for username, user_data in face_encoding_dict.items():
        # Handle both old format (direct encodings) and new format (dictionary with encoding)
        if isinstance(user_data, dict) and 'encoding' in user_data:
            known_encoding = user_data['encoding']
        else:
            # Old format where user_data is directly the encoding
            known_encoding = user_data
            
        distance = np.linalg.norm(known_encoding - unknown_encoding)
        
        if distance < best_distance:
            best_distance = distance
            best_match = username
    
    if best_distance > threshold:
        return "Intruder", None, None
    
    # Get the user data
    user_data = face_encoding_dict[best_match]
    
    # Handle both dictionary format and direct encoding format
    if isinstance(user_data, dict) and 'email' in user_data and 'password' in user_data:
        return best_match, user_data['email'], user_data['password']
    else:
        # Old format doesn't have email/password
        return best_match, None, None
def verify_face():
    """Face verification process that runs in a separate thread"""
    global is_verifying, verification_result, verification_start_time, recognized_name, recognized_email, recognized_password
    
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
                name, email, password = find_best_match(encoding)  # This line expects three return values
                
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
                        recognized_email = email
                        recognized_password = password
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
        'python_executable': r'D:/final-phase-project/FastSpeech/emailenv/Scripts/python.exe',
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
recognizer=None
try:
    if not os.path.exists("D:/final-phase-project/FastSpeech/vosk-model-small-en-us-0.15"):
        logger.error("Voice model not found. Please download the model from https://alphacephei.com/vosk/models")
    else:
        
        model = Model("D:/final-phase-project/FastSpeech/vosk-model-small-en-us-0.15")
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
    global recognizer
    
    # First, check if recognizer is None and try to initialize it if possible
    if recognizer is None:
        try:
            logger.info("Attempting to initialize voice recognizer...")
            model_path = "D:/final-phase-project/FastSpeech/vosk-model-small-en-us-0.15"
            if os.path.exists(model_path):
                model = Model(model_path)
                recognizer = KaldiRecognizer(model, 16000)
                logger.info("Voice recognizer initialized successfully")
            else:
                logger.error(f"Voice model not found at {model_path}")
                logger.error("Continuous listening cannot start. Please check the model path.")
                return
        except Exception as e:
            logger.error(f"Failed to initialize voice recognizer: {e}")
            logger.error("Continuous listening cannot start")
            return
    
    # Now proceed only if recognizer is properly initialized
    logger.info("Starting continuous listening...")
    
    try:
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
                if recognizer and recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result["text"]:
                        logger.info(f"Recognized: {result['text']}")
                        command_queue.put(result["text"].lower())
                        with credentials_lock:
                            user_email = user_credentials.get('email', EMAIL)
                            user_password = user_credentials.get('password', PASSWORD)
                        process_command(result["text"].lower(), user_email, user_password)

            except Exception as e:
                logger.error(f"Error in continuous listening loop: {e}")
                time.sleep(0.5)  # Short pause to prevent tight error loops
                continue

        logger.info("Continuous listening stopped")
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        logger.error(f"Fatal error in continuous listening: {e}")

def process_command(command, user_email=None, user_password=None):
    """Process voice commands for various actions"""
    logger.info(f"Processing command: {command}")
    
    # Email reading command
    if "read email" in command:
        logger.info("Processing 'read email' command...")
        if not user_email or not user_password:
            logger.error("Missing user credentials. Cannot fetch emails.")
            return False
        emails = fetch_unread_emails(email_address=user_email, password=user_password)
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
    
    elif "read archived emails" in command or "read read emails" in command or "read read email" in command:
        logger.info("Processing 'read archived emails' command...")
        if not user_email or not user_password:
            logger.error("Missing user credentials. Cannot fetch emails.")
            return False
        
        emails = fetch_read_emails(limit=5, email_address=user_email, password=user_password)
        if not emails:
            response_text = "No archived emails found."
        else:
            response_text = "Reading your archived emails:\n"
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
            
    elif "compose email" in command:
        logger.info("Processing 'compose email' command...")
        # Set command to handle in the UI
        command_queue.put("compose email")
        return True
        
    # Login related commands - we just pass these through to the UI
    elif any(keyword in command for keyword in ["manual", "username", "password", "face", "submit", "log in", "login now"]):
        logger.info(f"Login command detected: {command}")
        # The command will be handled by the JavaScript in the login page
        return True
        
    return False
@app.route('/synthesize_read', methods=['POST'])
def synthesize_read():
    if 'user' not in session:
        return jsonify({'error': 'User not authenticated'}), 403

    try:
        read_emails = fetch_read_emails(limit=5, email_address=session.get('email'), password=session.get('password'))
        if not read_emails:
            return jsonify({'error': 'No read emails to synthesize'})

        combined_text = "Reading your archived emails:\n\n"
        combined_text += "\n\n".join([
            f"From: {email['from'].split('<')[0].strip() if '<' in email['from'] else email['from']}. "
            f"Subject: {email['subject']}. Body: {email['body']}."
            for email in read_emails
        ])   

        file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_file = f"{file_id}.wav"
        output_file_path = os.path.join(OUTPUT_DIR, output_file)

        # Perform text-to-speech conversion
        text_to_speech(combined_text, output_file_path)

        # Verify file exists and has content
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            app.config['CURRENT_AUDIO'] = output_file
            # Return both success and the filename
            return jsonify({'success': True, 'audio_file': output_file})
        else:
            return jsonify({'error': 'Audio file not generated or is empty'})

    except Exception as e:
        logger.error(f"Read email synthesis error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/compose_email', methods=['GET', 'POST'])

def compose_email():
    """Email composition page"""
    if request.method == 'POST':
        recipient = request.form.get('recipient')
        subject = request.form.get('subject')
        body = request.form.get('body')
        
        if not recipient or not subject or not body:
            return render_template('compose_email.html', error="All fields are required")
        
        try:
            # Send the email
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = EMAIL
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL, PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL, recipient, text)
            server.quit()
            
            return redirect(url_for('home', success_message="Email sent successfully!"))
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return render_template('compose_email.html', recipient=recipient, 
                                  subject=subject, body=body, 
                                  error=f"Failed to send email: {str(e)}")
    
    return render_template('compose_email.html')
def start_listening_thread():
    thread = threading.Thread(target=continuous_listening)
    thread.daemon = True
    thread.start()

def fetch_unread_emails(mark_as_seen=False, email_address=None, password=None):
    """Fetch unread emails using provided credentials"""
    # Use provided credentials or fall back to defaults
    
    email_to_use = email_address or EMAIL
    password_to_use = password or PASSWORD
    
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_to_use, password_to_use)
        mail.select("inbox", readonly=not mark_as_seen)
        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()
        emails = []
        
        for email_id in reversed(email_ids):
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1] if isinstance(response_part[1], bytes) else response_part[1].encode())                    
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                    
                    from_ = msg.get("From")
                    
                    # Parse date with time information
                    date_str = msg.get("Date")
                    date_time = None
                    if date_str:
                        try:
                            date_time = parsedate_to_datetime(date_str)
                        except Exception as e:
                            logger.error(f"Error parsing date: {e}")
                    
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
                        "date": date_time.strftime("%Y-%m-%d") if date_time else "Unknown date",
                        "time": date_time.strftime("%H:%M:%S") if date_time else "Unknown time",
                        "datetime": date_time,
                        "body": body.strip() if body else "No body content"
                    })
                    
                    viewed_email_ids.add(email_id)
        
        mail.logout()
        return emails
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        

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
def fetch_read_emails(limit=10, email_address=None, password=None):
    """Fetch read emails using provided credentials, limited to a specific number"""
    
    email_to_use = email_address or EMAIL
    password_to_use = password or PASSWORD
    
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_to_use, password_to_use)
        mail.select("inbox", readonly=True)
        
        # Search for SEEN emails
        status, messages = mail.search(None, 'SEEN')
        email_ids = messages[0].split()
        
        # Limit the number of read emails we fetch (most recent first)
        if limit > 0 and len(email_ids) > limit:
            email_ids = email_ids[-limit:]
            
        emails = []
        
        for email_id in reversed(email_ids):
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1] if isinstance(response_part[1], bytes) else response_part[1].encode())
                    
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                    
                    from_ = msg.get("From")
                    
                    # Parse date with time information
                    date_str = msg.get("Date")
                    date_time = None
                    if date_str:
                        try:
                            date_time = parsedate_to_datetime(date_str)
                        except Exception as e:
                            logger.error(f"Error parsing date: {e}")
                    
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
                        "date": date_time.strftime("%Y-%m-%d") if date_time else "Unknown date",
                        "time": date_time.strftime("%H:%M:%S") if date_time else "Unknown time",
                        "datetime": date_time,
                        "body": body.strip() if body else "No body content"
                    })
        
        mail.logout()
        return emails
    except Exception as e:
        logger.error(f"Error fetching read emails: {e}")
        return []
@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Validate input
        if not username or not email or not password:
            return render_template('register_face.html', error="All fields are required")
            
        # Verify email credentials
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(email, password)
            mail.logout()
        except Exception as e:
            return render_template('register_face.html', 
                                  username=username,
                                  email=email,
                                  error=f"Email verification failed: {str(e)}")
        
        # Store in session for the face capture page
        session['reg_username'] = username
        session['reg_email'] = email
        session['reg_password'] = password
        
        return redirect(url_for('capture_face'))
        
    return render_template('register_face.html')

@app.route('/capture_face')
def capture_face():
    # Check if registration data exists in session
    if not all(key in session for key in ['reg_username', 'reg_email', 'reg_password']):
        return redirect(url_for('register_face'))
        
    return render_template('capture_face.html')

@app.route('/save_face', methods=['POST'])
def save_face():
    # Process the captured face image and save encoding
    if not all(key in session for key in ['reg_username', 'reg_email', 'reg_password']):
        return jsonify({'status': 'error', 'message': 'Registration data missing'})
    
    try:
        # Get the base64 image data from the request
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data received'})
            
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB for face detection
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detected_faces = detector.detect_faces(rgb_img)
        if not detected_faces:
            return jsonify({'status': 'error', 'message': 'No face detected in image'})
            
        # Use the largest face
        best_face = max(detected_faces, key=lambda f: f['box'][2] * f['box'][3])
        box = best_face['box']
        
        # Get face encoding
        face_encoding = get_face_encoding(rgb_img, box)
        
        # Save user data with face encoding
        username = session['reg_username']
        email = session['reg_email']
        password = session['reg_password']
        
        # Save the face encoding and credentials
        save_user_face(username, face_encoding, email, password)
        
        # Clear registration session data
        session.pop('reg_username', None)
        session.pop('reg_email', None)
        session.pop('reg_password', None)
        
        return jsonify({'status': 'success', 'message': 'Face registered successfully'})
    except Exception as e:
        logger.error(f"Error saving face: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})
@app.route('/check_verification')
def check_verification():
    """Check the status of face verification"""
    global is_verifying, verification_result, recognized_name, recognized_email, recognized_password
    
    if not is_verifying and verification_result:
        # Verification successful - use recognized name's email for login
        if recognized_name and recognized_name != "Intruder":
            # Set session variables with the recognized credentials
            session['user'] = recognized_name
            session['email'] = recognized_email 
            session['password'] = recognized_password
            
            # Also update the thread-safe credentials for background processes
            with credentials_lock:
                user_credentials['email'] = recognized_email
                user_credentials['password'] = recognized_password
            
            return {'status': 'success', 'name': recognized_name}
        else:
            # User exists in face recognition but not in system
            verification_result = False
            return {'status': 'failed', 'message': 'User not found in system'}
    elif not is_verifying and verification_result is False:
        # Verification failed
        return {'status': 'failed'}
    else:
        # Still verifying
        return {'status': 'verifying'}

@app.route('/logout')
def logout():
    """Logout route"""
    session.pop('user', None)
    session.pop('email', None)
    session.pop('password', None)
    return redirect(url_for('login'))


# -------------------------
# Main Application Routes
# -------------------------

@app.route('/')
def home():
    """Main application home page"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Use the user's email credentials to fetch emails
    user_email = session.get('email')
    user_password = session.get('password')
    
    try:
        # Attempt to fetch emails using the stored credentials
        unread_emails = fetch_unread_emails(email_address=user_email, password=user_password)
        read_emails = fetch_read_emails(limit=10, email_address=user_email, password=user_password)
        
        logger.debug(f"Fetched unread emails: {len(unread_emails)}, read emails: {len(read_emails)}")
        audio_file = app.config.get('CURRENT_AUDIO')
        
        # For backward compatibility with templates expecting 'emails' variable
        emails = unread_emails
        
        return render_template('index.html', 
                             emails=emails,  # Backward compatibility
                             unread_emails=unread_emails, 
                             read_emails=read_emails, 
                             audio_file=audio_file, 
                             username=session['user'])
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        # If there's an error fetching emails, it might be due to expired credentials
        # Clear session and redirect to login
        session.clear()
        return redirect(url_for('login'))
# Modify the face_encoding_dict structure to store user credentials along with face encodings
# Instead of:
# face_encoding_dict = {username: encoding}
# Use:
# face_encoding_dict = {username: {'encoding': encoding, 'email': email, 'password': password}}

def save_user_face(username, face_encoding, email, password):
    """Save user face encoding along with their email credentials"""
    global face_encoding_dict
    
    face_encoding_dict[username] = {
        'encoding': face_encoding,
        'email': email,
        'password': password
    }
    
    # Save to file
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(face_encoding_dict, f)
    
    logger.info(f"Saved face encoding and credentials for user: {username}")

@app.route('/audio/<filename>')
def serve_audio(filename):
    # Add debugging to verify the path
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        logger.info(f"Serving audio file: {file_path}")
        return send_from_directory(OUTPUT_DIR, filename, mimetype='audio/wav')
    else:
        logger.error(f"Audio file not found: {file_path}")
        return "Audio file not found", 404

@app.route('/check_commands')
def check_commands():
    try:
        # Non-blocking get from the queue with a timeout of 0.1 seconds
        command = command_queue.get(block=False)
        should_reload = "read email" in command  # Reload only if "read email" command detected
        redirect_compose = "compose email" in command 
        audio_file = app.config.get('CURRENT_AUDIO') if should_reload else None
        return jsonify({
            'command': command,
            'reload': should_reload,
            'compose_email': redirect_compose,
            'audio_file': audio_file
        })
    except queue.Empty:
        return jsonify({'command': None, 'reload': False, 'compose_email': False, 'audio_file':None})

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
            f"From: {email['from'].split('<')[0].strip() if '<' in email['from'] else email['from']}. "
            f"Subject: {email['subject']}. Body: {email['body']}."
            for email in emails
        ])   

        file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_file = f"{file_id}.wav"
        output_file_path = os.path.join(OUTPUT_DIR, output_file)

        # Perform text-to-speech conversion
        text_to_speech(combined_text, output_file_path)

        # Verify file exists and has content
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            app.config['CURRENT_AUDIO'] = output_file
            # Return both success and the filename
            return jsonify({'success': True, 'audio_file': output_file})
        else:
            return jsonify({'error': 'Audio file not generated or is empty'})

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

        python_bin = r'D:/final-phase-project/FastSpeech/emailenv/Scripts/python.exe'
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
    """Handle manual login by verifying email credentials"""
    email_address = request.form.get('email')
    password = request.form.get('password')
    
    # Validation
    if not email_address or not password:
        logger.warning("Login attempt with empty credentials")
        return render_template('login.html', error="Email and password are required")
    
    try:
        # Try to connect to email server and authenticate
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_address, password)
        
        # If we reach here, authentication was successful
        mail.logout()
        
        # Set session variables
        session['user'] = email_address
        session['email'] = email_address
        session['password'] = password  # Store securely for later email fetching
        
        # Also store in thread-safe structure for background threads
        with credentials_lock:
            user_credentials['email'] = email_address
            user_credentials['password'] = password
        
        logger.info(f"Successful login for user: {email_address}")
        return redirect(url_for('home'))
    except Exception as e:
        logger.warning(f"Failed login attempt for email: {email_address}, error: {str(e)}")
        return render_template('login.html', error="Invalid email or password. Please check your credentials.")

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
    login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Initialize database when app starts
with app.app_context():
    
    # Set default values
    app.config['CURRENT_AUDIO'] = None
    
    # Run the Flask app
    app.run(debug=True, use_reloader=False)