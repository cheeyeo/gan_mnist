# Builds a standard CNN model for evaluation against the generated images.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout

def cnn_model():
	model = Sequential()
	model.add(Conv2D(32, (5,5), padding="same", activation="relu", input_shape=(28, 28, 1)))
	model.add(Conv2D(32, (5,5), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
	model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation="softmax"))

	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

if __name__ == "__main__":
	model = cnn_model()
	model.summary()