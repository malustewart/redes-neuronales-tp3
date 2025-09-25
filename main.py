import numpy as np
import matplotlib.pyplot as plt

def plot_ISI_histogram(ISI, n_bins=40):
    fig, ax1 = plt.subplots()

    counts, bins, patches = ax1.hist(ISI, bins=n_bins, edgecolor="black", alpha=0.6)
    ax1.set_xlabel("Tiempo (ms)")
    ax1.set_ylabel("Frecuecia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    counts, bins, patches = ax2.hist(ISI, bins=n_bins, density=True, alpha=0)
    ax2.set_ylabel("Densidad de probabilidad", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Histograma de ISI con frecuencia (izquierda) y aproximación de densidad de probabilidad (derecha)")

neurons = np.loadtxt("dat/spikes.dat", dtype=int)
stimulus = np.loadtxt("dat/stimulus.dat", dtype=float)

## punto 1

ISI = np.concatenate([(np.diff(np.where(neuron == 1)).flatten()) for neuron in neurons])*0.1
plot_ISI_histogram(ISI=ISI)

ISI_avg = np.mean(ISI)
ISI_std = np.std(ISI)
CV = ISI_std/ISI_avg

print("ISI average: ", ISI_avg)
print("ISI std: ", ISI_std)
print("CV: ", CV)

### punto 2

# spike_count = np.

### punto 3

### punto 4

###
plt.show()