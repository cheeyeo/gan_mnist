import os
import gzip
import pickle
import numpy as np

from tensorflow.keras.utils import to_categorical

def load_real_samples():
	current_path = os.getcwd()

	file = "mnist.pkl.gz"
	path = os.path.sep.join([current_path, file])

	with gzip.open(path, "rb") as f:
		train_set, val_set, test_set = pickle.load(f, encoding="latin1")

	X_train, y_train = train_set
	X_val, y_val = val_set
	X_test, y_test = test_set

	print("X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))
	print("X_val shape: {}, y_val shape: {}".format(X_val.shape, y_val.shape))
	print("X_test shape: {}, y_test shape: {}".format(X_test.shape, y_test.shape))

	# Reshape data to be of shape (height, width, channel)
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
	X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

	# One-hot encode labels
	y_train = to_categorical(y_train)
	y_val = to_categorical(y_val)
	y_test = to_categorical(y_test)

	return X_train, y_train, X_val, y_val, X_test, y_test

def generate_real_samples(dataset, n):
	idx = np.random.randint(0, dataset.shape[0], n)

	X = dataset[idx]
	y = np.ones((n, 1))

	return X, y

def generate_latent_points(latent_dim, n):
	X = np.random.randn(latent_dim * n)
	X = X.reshape(n, latent_dim)
	return X

def generate_fake_samples(generator, latent_dim, n):
	x_input = generate_latent_points(latent_dim, n)

	X = generator.predict(x_input)
	y = np.zeros((n, 1))

	return X, y