from flask import Flask, request, render_template, send_from_directory
import os
import subprocess
import random
import imaplib
import time
import email
import atexit
import string
from email.header import decode_header

app = Flask(__name__)
EMAIL = "sreeragrok1@gmail.com"
PASSWORD = "lzpe wloy mzpc mzph"  # Use app-specific password if 2FA is enabled
viewed_email_ids = set()

OUTPUT_DIR = "./output/result/LJSpeech"
current_transcription = ""
current_audio_file = ""
last_transcription = ""
@app.route('/transcription', methods=['GET'])
def get_transcription():
    global current_transcription
    transcription = current_transcription
    current_transcription = ""  # Reset after reading
    return jsonify({"transcription": transcription})

# Function to fetch unread emails from Gmail
def fetch_unread_emails(mark_as_seen=False):
    # Connect to the Gmail IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")

    # Login to your Gmail account
    mail.login(EMAIL, PASSWORD)

    # Select the mailbox you want to check (e.g., 'inbox')
    mail.select("inbox", readonly=not mark_as_seen)

    # Search for unread emails using the 'UNSEEN' flag
    status, messages = mail.search(None, 'UNSEEN')  # Only fetch unread emails

    # Convert the result to a list of email IDs
    email_ids = messages[0].split()

    emails = []

    # Loop through each email ID and fetch the email content
    for email_id in reversed(email_ids):  # Reverse the order to get the latest email first
        # Fetch the email by ID
        status, msg_data = mail.fetch(email_id, "(RFC822)")

        for response_part in msg_data:
            if isinstance(response_part, tuple):
                # Parse the email content
                msg = email.message_from_bytes(response_part[1])

                # Decode the email subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")

                # Decode the sender's email address
                from_ = msg.get("From")

                # Extract email body
                body = None
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))

                        # Handle text/plain or text/html
                        if content_type in ["text/plain", "text/html"]:
                            payload = part.get_payload(decode=True)
                            if payload:  # Ensure there's a payload before decoding
                                body = payload.decode(errors='ignore')
                                if content_type == "text/plain":
                                    break  # Prefer plain text if available
                else:
                    # If the email is not multipart, get the payload
                    payload = msg.get_payload(decode=True)
                    if payload:  # Ensure there's a payload before decoding
                        body = payload.decode(errors='ignore')

                # Append email data to the list
                emails.append({
                    "subject": subject,
                    "from": from_,
                    "body": body.strip() if body else "No body content"
                })
                
                viewed_email_ids.add(email_id)

    # Logout from the email server
    mail.logout()

    return emails




@app.route('/')
def home():
    # Fetch unread emails when the home page loads
    emails = fetch_unread_emails()
    return render_template('index.html', emails=emails)


# Directory to save the generated audio
OUTPUT_DIR = "./output/result/LJSpeech"

@app.route('/audio/<filename>')
def serve_audio(filename):
    # Serve the audio file
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/check_audio/<filename>', methods=['GET'])
def check_audio(filename):
    audio_file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(audio_file_path):
        return jsonify({"status": "ready", "audio_file": filename})
    else:
        return jsonify({"status": "processing"})

@app.route('/synthesize', methods=['POST'])
def synthesize_text():
    emails = fetch_unread_emails()
    combined_text = "\n\n".join([
        f"From: {email['from'].split('<')[0].strip() if '<' in email['from'] else email['from']}."
        f"Subject: {email['subject']}."
        f"Body: {email['body']}."
        for email in emails
    ])   
    # Generate a unique identifier for the output file
    file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    output_file = f"{file_id}.wav"
    output_file_path = os.path.join(OUTPUT_DIR, output_file)

    # Specify the full path to the virtual environment's Python
    python_bin = r'D:\fast\FastSpeech2\newenv\Scripts\python.exe'

    # Command to run the synthesis script using the virtual environment's Python
    command = [
        python_bin, 'synthesize.py',
        '--text', combined_text,
        '--restore_step', '900000',  # Set the appropriate restore step
        '--mode', 'single',
        '--file_id', file_id,  # Pass the file_id to the synthesis script
        '-p', 'config/LJSpeech/preprocess.yaml',
        '-m', 'config/LJSpeech/model.yaml',
        '-t', 'config/LJSpeech/train.yaml'
    ]
    
    try:
        # Run the synthesis command and capture the output
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)  # Print stdout for debugging
        print(result.stderr)  # Print stderr for debugging
        
        # Return the result with the audio file path
        return render_template('index.html', audio_file=output_file)

    except subprocess.CalledProcessError as e:
        # Log the error message for debugging
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        return f"Error: {e.stderr}"




@app.route('/audio/<filename>')
def serve_audio(filename):
    # Serve the audio file
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/synthesize_voice', methods=['POST'])
def synthesize_voice():
    global current_audio_file
    emails = fetch_unread_emails()
    combined_text = "\n\n".join([
        f"From: {email['from']}.\nSubject: {email['subject']}.\nBody: {email['body']}"
        for email in emails
    ])    
    
    file_id = str(int(time.time())) + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    output_file = f"{file_id}.wav"
    output_file_path = os.path.join(OUTPUT_DIR, output_file)

    python_bin = r'D:\fast\FastSpeech2\newenv\Scripts\python.exe'
    command = [
        python_bin, 'synthesize.py',
        '--text', combined_text,
        '--restore_step', '900000',
        '--mode', 'single',
        '--file_id', file_id,
        '-p', 'config/LJSpeech/preprocess.yaml',
        '-m', 'config/LJSpeech/model.yaml',
        '-t', 'config/LJSpeech/train.yaml'
    ]
    
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        current_audio_file = output_file
        return jsonify({
            "status": "success",
            "audio_file": output_file
        })
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        return jsonify({
            "status": "error",
            "message": str(e.stderr)
        })

def mark_as_read_on_exit():
    # Connect to the Gmail IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    
    # Login to your Gmail account
    mail.login(EMAIL, PASSWORD)
    
    # Select the mailbox you want to check (e.g., 'inbox')
    mail.select("inbox")
    
    for email_id in viewed_email_ids:
        # Mark email as seen
        mail.store(email_id, '+FLAGS', '\\Seen')
    
    # Logout from the email server
    mail.logout()


# Register the exit handler to mark emails as read on shutdown
atexit.register(mark_as_read_on_exit)


if __name__ == '__main__':
    app.run(debug=True)

