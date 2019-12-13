# Trains the cnn model defined in cnn_model.py

from data import load_real_samples
from cnn_model import cnn_model
from utils import plot_history

# Loading the MNIST dataset
X_train, y_train, X_val, y_val, _, _ = load_real_samples()
print("[INFO] X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))
print("[INFO] X_val shape: {}, y_val shape: {}".format(X_val.shape, y_val.shape))

print("[INFO] Training model...")
model = cnn_model()
H = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

plot_history(H, 100)

model.save("models/cnn_model.h5")