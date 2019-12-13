# Trains GAN model defined in gan_model.py
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
from data import load_real_samples
from data import generate_real_samples
from data import generate_fake_samples
from data import generate_latent_points
from gan_model import define_generator, define_discriminator, define_gan_model

gmodel = define_generator()
dmodel = define_discriminator()
gan_model = define_gan_model(gmodel, dmodel)

dataset, _, _, _, _, _ = load_real_samples()
print(dataset.shape)

batch = 256
# epochs = 2000
epochs = 50
latent_dim = 100

half_batch = int(batch / 2)
batch_per_epoch = int(dataset.shape[0] / batch)

steps = batch_per_epoch * epochs
print("[INFO] BATCH PER EPOCH: {}".format(batch_per_epoch))
print("[INFO] STEPS: {}".format(steps))

d1_hist = list()
d2_hist = list()
g_hist = list()
a1_hist = list()
a2_hist = list()

for i in range(steps):
	X_real, y_real = generate_real_samples(dataset, half_batch)

	d_loss1, d_acc1 = dmodel.train_on_batch(X_real, y_real)

	X_fake, y_fake = generate_fake_samples(gmodel, latent_dim, half_batch)

	d_loss2, d_acc2 = dmodel.train_on_batch(X_fake, y_fake)

	X_gan = generate_latent_points(latent_dim, batch)
	y_gan = np.ones((batch, 1))

	g_loss = gan_model.train_on_batch(X_gan, y_gan)

	print("[INFO] Epoch: {:d}, d1={:.3f}, d2={:.3f}, g={:.3f}, a1={:d}, a2={:d}".format(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))

	d1_hist.append(d_loss1)
	d2_hist.append(d_loss2)
	g_hist.append(g_loss)
	a1_hist.append(d_acc1)
	a2_hist.append(d_acc2)

	if (i+1) % batch_per_epoch == 0:
		print(i)
		print("SAVING artifacts!!!")
		X, _ = generate_fake_samples(gmodel, latent_dim, 16)

		for j in range(4*4):
			plt.subplot(4, 4, j+1)
			plt.axis("off")
			plt.imshow(X[j, :, :, 0], cmap="gray")

		plt.savefig("artifacts/generated_plot_{:03d}.png".format(i+1))
		plt.close()

		gmodel.save("models/model_{:03d}.h5".format(i+1))

	# TODO: Add plot of loss/acc for evaluation