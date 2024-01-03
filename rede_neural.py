from tensorflow.python.compiler.tensorrt import trt_convert as trt
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import sys
import os

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = 'Database_134_Angiograms/train/'
TEST_PATH = 'Database_134_Angiograms/valid/'

train_ids = next(os.walk(TRAIN_PATH + 'images/'))[2]
test_ids = next(os.walk(TEST_PATH + 'images/'))[2]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

print("Resizing training images and masks")
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(TRAIN_PATH + 'images/' + id_)  # Lê a imagem original
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    img = np.concatenate([img, img, img], axis=-1)  # Replica para 3 canais
    X_train[n] = img
    
    mask_id = id_[:-4] + '_gt.pgm'
    mask = imread(TRAIN_PATH + 'masks/' + mask_id)
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    Y_train[n] = mask

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
size_test = []
print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(TEST_PATH + 'images/' + id_)
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    img = np.concatenate([img, img, img], axis=-1)  # Replica para 3 canais
    X_test[n] = img
print("Done!")

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

# Defina uma métrica personalizada de IoU (Intersection over Union)
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.float32)
        intersection = K.sum(K.abs(y_true * y_pred_), axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred_, axis=[1, 2, 3]) - intersection
        prec.append((intersection + K.epsilon()) / (union + K.epsilon()))
    return K.mean(K.stack(prec), axis=0)

# Divider data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Build the model

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Primary layer
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

# Second layer
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
# Third layer
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
# Fourth layer
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
# Fifth layer
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',  mean_iou])
model.summary()

# Adicione quantização ao modelo


#########################################################################################################
########################################## ModelCheckpoint ##############################################
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

##########################################################################################################

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Verify a sanity check on same on random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()

# Verify a sanity check on same on random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

# Choose a color for overlay (e.g., green)
overlay_color = [0, 255, 0]  # RGB values

# Choose a threshold for the predicted mask
threshold = 0.5

# Helper function to overlay mask on image
# Helper function to overlay mask on image
# Helper function to overlay mask on image
# Helper function to overlay mask on image
def overlay_mask(image, mask, color, threshold=0.5):
    overlay = image.copy()
    mask = (mask > threshold).astype(np.uint8)
    mask = np.repeat(mask, 3, axis=-1)  # Repeat the mask for each channel
    overlay[mask == 1] = color
    return overlay

# Choose a random index for visualization
idx = random.randint(0, len(preds_val_t))

# Overlay the predicted mask on the original image
original_image = X_train[int(X_train.shape[0]*0.9):][idx]
predicted_mask = preds_val_t[idx]
overlay_image = overlay_mask(original_image, predicted_mask, overlay_color, threshold)

# Visualize the result
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(predicted_mask), cmap='gray')
plt.title('Predicted Mask')

plt.subplot(1, 3, 3)
plt.imshow(overlay_image)
plt.title('Overlay Image')

plt.show()

# Durante a avaliação, obtenha as métricas
loss, accuracy, mean_iou_value = model.evaluate(X_test, Y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}, Mean IoU: {mean_iou_value}')


# Convert the Keras model to a TensorFlow SavedModel
model.save("keras_model")

# Convert the TensorFlow SavedModel to a TensorRT model
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode="FP16", max_workspace_size_bytes=(1<<30))
converter = trt.TrtGraphConverterV2(input_saved_model_dir="keras_model", conversion_params=conversion_params)
converter.convert()
converter.save("tensorrt_model")

# Load the TensorRT model
trt_model = tf.saved_model.load("tensorrt_model")

# Compile the TensorRT model
trt_compiled_model = tf.function(trt_model)

# Use the compiled TensorRT model for inference
preds_train = trt_compiled_model(X_train[:int(X_train.shape[0]*0.9)], training=False)
preds_val = trt_compiled_model(X_train[int(X_train.shape[0]*0.9):], training=False)
preds_test = trt_compiled_model(X_test, training=False)