import pickle
import os
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)
print(data.keys())  # Should list registered usernames
if not os.path.exists("models/shape_predictor_68_face_landmarks.dat") or not os.path.exists("models/dlib_face_recognition_resnet_model_v1.dat"):
    print("Dlib models are missing!")
