# Defines the generator, discriminator, GAN models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop

def define_generator(depth=256, dim=7, dropout=0.3, momentum=0.8, latent_dim=100):

	model = Sequential()

	nodes = depth*dim*dim
	model.add(Dense(nodes, input_dim=latent_dim))
	model.add(BatchNormalization(momentum=momentum))
	model.add(Activation("relu"))
	model.add(Reshape((dim, dim, depth)))
	model.add(Dropout(dropout))

	# Upsample to 14x14x128
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(BatchNormalization(momentum=momentum))
	model.add(Activation("relu"))

	# Upsample to 28x28x64
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding="same"))
	model.add(BatchNormalization(momentum=momentum))
	model.add(Activation("relu"))

	# Upsample to 28x28x32
	model.add(Conv2DTranspose(32, (4,4), padding="same"))
	model.add(BatchNormalization(momentum=momentum))
	model.add(Activation("relu"))

	model.add(Conv2DTranspose(1, (4,4), padding="same"))
	model.add(Activation("sigmoid"))

	return model

# 28 x 28 x 1 → 14 x 14 x 64 → 7 x 7 x 128 → 4 x 4 x 256 → 2 x 2 x 512 → 1
def define_discriminator(depth=64, dropout=0.3, alpha=0.3, input_shape=(28, 28, 1)):

	model = Sequential()

	# Downsample to 14x14x64
	model.add(Conv2D(64, (4,4), strides=(2,2), padding="same", input_shape=input_shape))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Dropout(dropout))

	# Downsample to 7x7x128
	model.add(Conv2D(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Dropout(dropout))

	# Downsample to 4x4x256
	model.add(Conv2D(256, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Dropout(dropout))

	# Downsample to 2x2x512
	model.add(Conv2D(512, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(1))
	model.add(Activation("sigmoid"))

	opt = RMSprop(lr=0.0002, decay=6e-8)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	return model

def define_gan_model(generator, discriminator):
	discriminator.trainable = False

	model = Sequential()
	model.add(generator)
	model.add(discriminator)

	opt = RMSprop(lr=0.0001, decay=3e-8)
	model.compile(loss="binary_crossentropy", optimizer=opt)

	return model


if __name__ == "__main__":
	gmodel = define_generator()
	# gmodel.summary()

	dmodel = define_discriminator()
	# dmodel.summary()

	gan_model = define_gan_model(gmodel, dmodel)
	gan_model.summary()