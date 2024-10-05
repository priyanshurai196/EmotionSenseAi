import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('model_file_30epochs.h5')

# Start video capture
video = cv2.VideoCapture(0)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'sad', 6: 'surprise'}

# Define project name to display
project_name = "EmotionSenseAI"

while True:
    # Capture frame from video
    ret, frame = video.read()
    
    if not ret:
        print("Failed to capture video frame")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Display the project name at the top of the frame
    cv2.putText(frame, project_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for (x, y, w, h) in faces:
        # Extract the region of interest (the face)
        face = gray_frame[y:y+h, x:x+w]
        
        # Resize the face to the model's input size (48x48)
        resized_face = cv2.resize(face, (48, 48))
        
        # Normalize the pixel values (0-255 to 0-1)
        normalized_face = resized_face / 255.0
        
        # Reshape the image to match the model input shape (1, 48, 48, 1)
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
        
        # Predict the emotion
        result = model.predict(reshaped_face)
        label_index = np.argmax(result, axis=1)[0]
        label = labels_dict[label_index]
        
        # Draw rectangles around the face and display the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display the frame with annotations
    cv2.imshow("Emotion Detection", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
video.release()
cv2.destroyAllWindows()
