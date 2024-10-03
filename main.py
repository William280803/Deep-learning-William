import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Plot accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss Curves using model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy history
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, len(model_history.history['accuracy']) / 10))
    axs[0].legend(['train', 'val'], loc='best')
    
    # Plot loss history
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss']) / 10))
    axs[1].legend(['train', 'val'], loc='best')
    
    fig.savefig('plot.png')
    plt.show()

# Data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 1

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

# Building the model
model = Sequential()
model.add(Input(shape=(48, 48, 1)))  # Input layer
model.add(Conv2D(32, (3, 3), activation='relu'))  # First Conv layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # First MaxPooling layer
model.add(Conv2D(64, (3, 3), activation='relu'))  # Second Conv layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Second MaxPooling layer
model.add(Conv2D(128, (3, 3), activation='relu'))  # Third Conv layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Third MaxPooling layer
model.add(Flatten())  # Flatten layer
model.add(Dense(7, activation='softmax'))  # Output layer

# Training mode
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size
    )
    plot_model_history(model_info)
    model.save_weights('model.h5')  # Save weights after training

# Display mode
elif mode == "display":
    # print("Current model architecture:")
    # model.summary()  # Print the model architecture

    # if os.path.exists('model.h5'):
    #     try:
    #         model.load_weights('model.h5')
    #         print("Weights loaded successfully.")
    #     except ValueError as e:
    #         print(f"Error loading weights: {e}")
    # else:
    #     print("Model weights file not found. Please train the model first.")

    model.load_weights('model.h5')

    # Prevent OpenCL usage and unnecessary logging
    cv2.ocl.setUseOpenCL(False)

    # Emotion dictionary
    emotion_dict = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Netral", 5: "Sedih", 6: "Terkejut"}

    # Start webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Output', cv2.resize(frame, (1000, 600), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
