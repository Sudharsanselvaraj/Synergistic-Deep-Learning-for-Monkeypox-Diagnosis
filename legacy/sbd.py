import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import json
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load and preprocess data
data = pd.read_csv("C:/Users/vviji/OneDrive/Desktop/DATA.csv")
data = data.drop(columns='Patient_ID')
data = data.dropna()

# Encoding categorical data
column_to_encode = 'Systemic Illness'
encoded_data = pd.get_dummies(data[column_to_encode], prefix=column_to_encode)
data = pd.concat([data, encoded_data], axis=1)
data = data.drop(column_to_encode, axis=1)

# Mapping values
mapping = {
    True: 1,
    False: 0,
    "Positive": 1,
    "Negative": 0
}
data = data.applymap(lambda x: mapping.get(x, x))

# Analyzing correlation
df = data.copy()
df['sum'] = df.sum(axis=1)

# Splitting data
model_features = df.columns.tolist()
model_features.remove('MonkeyPox')  # Assuming 'MonkeyPox' is the target

# Save feature names to a JSON file
with open("model_features.json", "w") as f:
    json.dump(model_features, f)
print("Feature names saved to model_features.json")

X = df[model_features]
y = df['MonkeyPox']

# Stratified splitting to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
X_eval, X_test, y_eval, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.save")
print("Scaler saved as scaler.save")

# Compute class weights to handle potential class imbalance
class_weights_values = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_values))
print("Class weights computed:", class_weights_dict)

# Convert to tensors
X_train_tensor = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
X_eval_tensor = tf.convert_to_tensor(X_eval_scaled, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
y_eval_tensor = tf.convert_to_tensor(y_eval.values, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Model training
with tf.device('/GPU:0'):
    model = keras.Sequential([
        layers.Input(shape=(X_train_tensor.shape[1],)),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, mode='max', restore_best_weights=True)

    # Capture the training history
    history = model.fit(
        X_train_tensor, y_train_tensor,
        epochs=70,
        callbacks=[early_stopping],
        batch_size=32,
        validation_data=(X_eval_tensor, y_eval_tensor),
        class_weight=class_weights_dict
    )

# Save the trained model
model.save("MPOXSBD.h5")
print("MPOXSBD.keras")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_tensor, y_test_tensor, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()