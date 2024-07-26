import cv2
import dlib
import numpy as np
import os
import pickle
import threading
import pygame


class FaceRecognizerThread(threading.Thread):
    def __init__(self, known_faces, known_names):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.face_rec_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
        self.known_faces = known_faces
        self.known_names = known_names
        self.frame = None
        self.result = None
        self.lock = threading.Lock()
        self.running = True
        self.playing_sound = False

        pygame.mixer.init()

    def run(self):
        while self.running:
            if self.frame is not None:
                with self.lock:
                    rgb_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    rgb_img = np.array(rgb_img, dtype=np.uint8)
                    faces = self.detector(rgb_img, 1)
                    results = []

                    for face in faces:
                        shape = self.sp(rgb_img, face)
                        face_descriptor = self.face_rec_model.compute_face_descriptor(rgb_img, shape)
                        face_descriptor = np.array(face_descriptor)
                        
                        distances = [np.linalg.norm(face_descriptor - known_face) for known_face in self.known_faces]
                        min_distance_index = distances.index(min(distances))
                        name = self.known_names[min_distance_index] if distances[min_distance_index] < 0.5 else "Unknown"
                        
                        if name == "CR7" and not self.playing_sound:
                            self.playing_sound = True
                            threading.Thread(target=self.play_sound, args=('audio_files/siu.mp3',)).start() 
                          
                        if name == "Messi" and not self.playing_sound:
                            self.playing_sound = True
                            threading.Thread(target=self.play_sound, args=('audio_files/ankara_messi.mp3',)).start() 
                            
                        if name == "MoSalah" and not self.playing_sound:
                            self.playing_sound = True
                            threading.Thread(target=self.play_sound, args=('audio_files/mo_salah.mp3',)).start() 
                            

                        results.append((face, name))
                    
                    self.result = results
                    self.frame = None

    def play_sound(self, sound_file):
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        self.playing_sound = False

    def recognize(self, frame):
        with self.lock:
            self.frame = frame

    def get_result(self):
        return self.result

    def stop(self):
        self.running = False

def load_known_faces(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            known_faces, known_names = pickle.load(f)
        print(f"Loaded known faces and names from {filename}")
        return known_faces, known_names
    else:
        print(f"No saved data found. You may need to train the model.")
        return [], []

def recognize_faces_in_webcam(known_faces, known_names):
    video_capture = cv2.VideoCapture(0)
    face_recognizer_thread = FaceRecognizerThread(known_faces, known_names)
    face_recognizer_thread.start()

    while True:
        _, img = video_capture.read()

        face_recognizer_thread.recognize(img)
        results = face_recognizer_thread.get_result()

        if results is not None:
            for face, name in results:
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_recognizer_thread.stop()
    face_recognizer_thread.join()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    known_faces_dir = os.path.join(base_dir, '..', 'known_faces')
    saved_faces_file = os.path.join(base_dir, '../processed_data.pkl')


    known_faces, known_names = load_known_faces(saved_faces_file)
    recognize_faces_in_webcam(known_faces, known_names)
