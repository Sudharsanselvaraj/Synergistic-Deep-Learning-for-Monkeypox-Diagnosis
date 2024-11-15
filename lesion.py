# Import necessary libraries
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB4, InceptionResNetV2, DenseNet201
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Define dataset directories (update these paths)
train_directory = 'C:/Users/vviji/Downloads/MPOXSKLDATASET/train'  # Update this path
validation_directory = 'C:/Users/vviji/Downloads/MPOXSKLDATASET/val'  # Update this path
test_directory = 'C:/Users/vviji/Downloads/MPOXSKLDATASET/test'  # Update this path

# 1. Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 14  # Update as per your dataset
EPOCHS = 50

# 2. Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = test_val_datagen.flow_from_directory(
    validation_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    test_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 3. Class Weights
classes = train_generator.classes
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights))

# 4. Base Model Creation
def create_base_model(model_name, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    if model_name == 'EfficientNetB4':
        return EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'InceptionResNetV2':
        return InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'DenseNet201':
        return DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model name")

base_model1 = create_base_model('EfficientNetB4')
base_model2 = create_base_model('InceptionResNetV2')
base_model3 = create_base_model('DenseNet201')

# 5. Building the Ensemble Model
inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

x1 = base_model1(inputs)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.Dropout(0.5)(x1)

x2 = base_model2(inputs)
x2 = layers.GlobalAveragePooling2D()(x2)
x2 = layers.Dropout(0.5)(x2)

x3 = base_model3(inputs)
x3 = layers.GlobalAveragePooling2D()(x3)
x3 = layers.Dropout(0.5)(x3)

concatenated = layers.Concatenate()([x1, x2, x3])
fc = layers.Dense(1024, activation='relu')(concatenated)
fc = layers.Dropout(0.5)(fc)
fc = layers.Dense(512, activation='relu')(fc)
fc = layers.Dropout(0.5)(fc)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(fc)

model = Model(inputs=inputs, outputs=outputs)

# 6. Compile the Model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
checkpoint = ModelCheckpoint('MPOXSKL.keras', monitor='val_loss', save_best_only=True)

# 8. Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
# Save the entire model after training
model.save('MPOXSKL.keras')
print("Model saved as 'MPOXSKL.h5'")

# 9. Fine-Tuning
for base_model in [base_model1, base_model2, base_model3]:
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# 10. Evaluate the Model
model.load_weights('MPOXSKL.keras')
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Predictions
Y_pred = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Classification Report
print('Classification Report')
target_names = list(train_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
