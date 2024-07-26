import cv2
import dlib
import numpy as np
import os
import pickle


def save_known_faces(known_faces, known_names, filename):
    with open(filename, 'wb') as f:
        pickle.dump((known_faces, known_names), f)
    print(f"Saved known faces and names to {filename}")



def process_faces(known_faces_dir, filename):
    known_faces = []
    known_names = []

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    face_rec_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Error loading image: {img_path}")
                continue

            # Convert to RGB and ensure it's uint8
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = np.array(image_rgb, dtype=np.uint8)

            # Debugging prints
            print(f"Processing image: {img_path}")
            print(f"Image shape: {image_rgb.shape}")
            print(f"Image dtype: {image_rgb.dtype}")

            # Ensure the image data is correct
            if image_rgb.max() > 255 or image_rgb.min() < 0:
                print(f"Error: The image has values outside the 8-bit range.")
                continue

            faces = detector(image_rgb, 1)

            for face in faces:
                shape = sp(image_rgb, face)
                face_descriptor = face_rec_model.compute_face_descriptor(image_rgb, shape)
                known_faces.append(np.array(face_descriptor))
                known_names.append(person_name)
    
    save_known_faces(known_faces, known_names, filename)



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    known_faces_dir = os.path.join(base_dir, '..', 'known_faces')
    saved_faces_file = os.path.join(base_dir, '../processed_data.pkl')
    
    process_faces(known_faces_dir, saved_faces_file)