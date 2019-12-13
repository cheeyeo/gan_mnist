import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, epochs):
	N = np.arange(0, epochs)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, history.history["loss"], label="train_loss")
	plt.plot(N, history.history["val_loss"], label="val_loss")
	plt.plot(N, history.history["accuracy"], label="train_acc")
	plt.plot(N, history.history["val_accuracy"], label="val_acc")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/accuracy")
	plt.legend(loc="lower left")
	plt.savefig("artifacts/cnn_training_plot.png")