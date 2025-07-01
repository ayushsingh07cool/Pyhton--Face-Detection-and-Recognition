import cv2
import os
import numpy as np

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare training data
labels = []
faces = []
label_dict = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue  # Skip files

    label_dict[current_id] = person_name  # Assign ID to name

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if face_img is None:
            continue  # Skip corrupted images
        
        faces.append(face_img)
        labels.append(current_id)

    current_id += 1

# Train the recognizer
recognizer.train(faces, np.array(labels))
recognizer.save("trained_model.yml")  # Save the trained model

# Save label dictionary
np.save("label_dict.npy", label_dict)

print("Training completed successfully!")
