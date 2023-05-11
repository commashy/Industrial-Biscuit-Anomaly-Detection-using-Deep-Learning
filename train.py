import os
import numpy as np
from PIL import Image
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
import efficientnet.keras as effnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.keras import TqdmCallback
import pandas as pd

def load_data(dsPath, folder):
    x = []
    y = []

    labels = {'ok': 0, 'nok': 1}

    for label in ['ok', 'nok']:
        path = os.path.join(dsPath, folder, label)
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                im = Image.open(os.path.join(path, file))
                im_array = np.array(im)
                x.append(im_array)
                y.append(labels[label])

    x = np.array(x)
    y = np.array(y)

    return x, y

# Set the dataset path
dsPath = '../IndustryBiscuit_Folders'

# Load the dataset
x_train, y_train = load_data(dsPath, 'train')
x_val, y_val = load_data(dsPath, 'valid')
x_test, y_test = load_data(dsPath, 'test')

# Preprocess the input data
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

data_gen.fit(x_train)

# Create the Xception model
# def create_model(input_shape, num_classes):
#     base_model = Xception(include_top=False, input_shape=input_shape, weights='imagenet')

#     inputs = layers.Input(shape=input_shape)
#     x = base_model(inputs)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs=inputs, outputs=outputs)

#     model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# Create the EfficientNetB0 model
# def create_model(input_shape, num_classes):
#     base_model = effnet.EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')

#     inputs = layers.Input(shape=input_shape)
#     x = base_model(inputs)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs=inputs, outputs=outputs)

#     model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])
#     return model

# Create the EfficientNetB4 model
def create_model(input_shape, num_classes):
    base_model = effnet.EfficientNetB4(include_top=False, input_shape=input_shape, weights='imagenet')

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

input_shape = (256, 256, 3)
num_classes = 4  # Number of classes in the dataset

model = create_model(input_shape, num_classes)

input_shape = (256, 256, 3)
num_classes = 4  # Number of classes in the dataset

model = create_model(input_shape, num_classes)

# Train the model
epochs = 30
batch_size = 32
history = model.fit(
    data_gen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=0,
    callbacks=[TqdmCallback(verbose=1)],
)

# Save the training progress to a CSV file
training_progress = pd.DataFrame(history.history)
training_progress['epoch'] = np.arange(1, epochs + 1)
training_progress = training_progress[['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy']]
training_progress.to_csv("training_progress.csv", index=False)

# Save the model
model.save("my_model.h5")

# Load the pretrained model
# pretrained_model_path = 'EfficientNetB0.h5'
# pretrained_model = model.load_weights(pretrained_model_path)

# Evaluate the model
y_pred = np.argmax(model.predict(x_test), axis=-1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
