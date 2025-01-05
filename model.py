import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# Define the U-Net model architecture with a classification head
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    # Classification head
    flat = Flatten()(conv10)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[inputs], outputs=[conv10, dense2])
    return model

# Load and preprocess the dataset
def load_dataset(dataset_path):
    images = []
    masks = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg') and not filename.endswith('_mask.jpg'):
            img_path = os.path.join(dataset_path, filename)
            mask_path = os.path.join(dataset_path, filename.replace('.jpg', '_mask.png'))
            if not os.path.exists(mask_path):
                print(f"Mask not found for {filename}")
                continue
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"Failed to read image or mask for {filename}")
                continue
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            img = img / 255.0
            mask = mask / 255.0
            images.append(img)
            masks.append(mask)
            # Load corresponding label (0 for bad, 1 for good)
            label = int(filename.split('_')[-1].split('.')[0])  # Assuming filename format: image_0.jpg or image_1.jpg
            labels.append(label)
    print(f"Loaded {len(images)} images, {len(masks)} masks, and {len(labels)} labels.")
    return np.array(images), np.array(masks), np.array(labels).reshape(-1, 1)

# Define the new loss function
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1) / (denominator + 1)

def combined_loss(y_true, y_pred):
    y_true_seg, y_true_cls = y_true
    y_pred_seg, y_pred_cls = y_pred
    d_loss = dice_loss(y_true_seg, y_pred_seg)
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true_cls, y_pred_cls)
    return d_loss + bce_loss

# Train the model
def train_model(model, dataset_path, epochs=50, batch_size=8):
    steps_per_epoch = len([f for f in os.listdir(dataset_path) if f.endswith('.jpg') and not f.endswith('_mask.jpg')]) // batch_size
    model.compile(optimizer=Adam(), loss=dice_loss, metrics=["accuracy", "binary_crossentropy"])
  
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(dataset_path, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
            (tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32), tf.TensorSpec(shape=(None, 1), dtype=tf.float32))
        )
    )
    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=dataset.take(10).batch(batch_size),  # Use a portion for validation
    )
    model.save('unet_model_with_classification.h5')

# Data generator for efficient loading
def data_generator(dataset_path, batch_size=8):
    file_list = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') and not f.endswith('_mask.jpg')]
    while True:
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i + batch_size]
            images, masks, labels = [], [], []
            for filename in batch_files:
                img_path = os.path.join(dataset_path, filename)
                mask_path = os.path.join(dataset_path, filename.replace('.jpg', '_mask.png'))
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (256, 256)) / 255.0
                mask = cv2.resize(mask, (256, 256)) / 255.0
                label = int(filename.split('_')[-1].split('.')[0])
                images.append(img)
                masks.append(mask)
                labels.append(label)
            yield np.array(images), [np.array(masks).reshape(-1, 256, 256, 1), np.array(labels).reshape(-1, 1)]

# Function to break videos into frames and save masks
def break_video_into_frames(video_folder, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subfolder in os.listdir(video_folder):
        subfolder_path = os.path.join(video_folder, subfolder)
        print(f"Processing videos in {subfolder_path}")
        if os.path.isdir(subfolder_path):
            frame_counter = 1
            for video_filename in os.listdir(subfolder_path):
                if video_filename.endswith('.mp4'):
                    video_path = os.path.join(subfolder_path, video_filename)
                    print(f"Processing video {video_path}")
                    video = cv2.VideoCapture(video_path)
                    if not video.isOpened():
                        print(f"Failed to open video {video_path}")
                        continue
                    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
                    count = 0
                    while video.isOpened():
                        ret, frame = video.read()
                        if not ret or frame is None:
                            print(f"Failed to read frame from {video_path}")
                            break
                        if count % frame_rate == 0:  # Save one frame per second
                            frame_filename = f"{subfolder}_{frame_counter}.jpg"
                            mask_filename = f"{subfolder}_{frame_counter}_mask.png"
                            cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
                            # Create a dummy mask for demonstration purposes
                            mask = np.zeros_like(frame[:, :, 0])
                            cv2.imwrite(os.path.join(output_dir, mask_filename), mask)
                            print(f"Saved frame {frame_filename} and mask {mask_filename}")
                            frame_counter += 1
                        count += 1
                    video.release()
    print(f"Frames and masks saved in {output_dir}")

# Main script
video_folder = "F:\\Mining\\BlastCaptain\\Data1of3"
dataset_path = 'dataset'

# Check if dataset folder exists, if not create it
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

break_video_into_frames(video_folder, dataset_path)
images, masks, labels = load_dataset(dataset_path)
input_size = (256, 256, 3)
model = unet_model(input_size)
train_model(model, dataset_path)