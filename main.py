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

def plot_spike_count_histogram(spike_count, n_bins=10):
    fig, ax1 = plt.subplots()

    counts, bins, patches = ax1.hist(spike_count, bins=n_bins, edgecolor="black", alpha=0.6)
    ax1.set_xlabel("Tiempo (ms)")
    ax1.set_ylabel("Frecuecia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    counts, bins, patches = ax2.hist(spike_count, bins=n_bins, density=True, alpha=0)
    ax2.set_ylabel("Densidad de probabilidad", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Histograma de cantidad de spikes por muestra con frecuencia (izquierda) y aproximación de densidad de probabilidad (derecha)")

def plot_stimulus(stimulus):
    plt.figure()
    plt.plot(stimulus[:,0],stimulus[:, 1])

def plot_raster_plot_and_spike_counts(t_spikes, spike_counts):
    fig, axs = plt.subplots(2)
    axs[0].plot(t_samples, spike_counts)

    axs[1].eventplot(t_spikes)

neurons = np.loadtxt("dat/spikes.dat", dtype=int)
stimulus = np.loadtxt("dat/stimulus.dat", dtype=float)

## punto 1

ISI = np.concatenate([(np.diff(np.where(neuron == 1)).flatten()) for neuron in neurons])*0.1
# plot_ISI_histogram(ISI=ISI)

ISI_avg = np.mean(ISI)
ISI_std = np.std(ISI)
CV = ISI_std/ISI_avg

print("ISI average: ", ISI_avg)
print("ISI std: ", ISI_std)
print("CV: ", CV)

### punto 2

spike_count = [np.count_nonzero(neuron) for neuron in neurons]

spike_count_avg = np.mean(spike_count)
spike_count_std = np.std(spike_count)
FF = spike_count_std**2/spike_count_avg

print("Spike count average: ", spike_count_avg)
print("Spike count std: ", spike_count_std)
print("FF: ", FF)

# plot_spike_count_histogram(spike_count, n_bins=10)
# plot_spike_count_histogram(spike_count, n_bins=20)
# plot_spike_count_histogram(spike_count, n_bins=30)


### punto 3


n_samples = len(neurons[0])
n_neurons = len(neurons)
t_samples = np.arange(len(neurons[0])) * 0.1
binlen = 25
step = 1
windows_start = range(0, n_samples, step)


spike_counts = np.repeat([np.count_nonzero(neurons[:, window_start:window_start+binlen]) for window_start in windows_start], step) / binlen / n_neurons
t_spikes = [np.where(neuron==1)[0] for neuron in neurons]

print(len(spike_counts))

plot_raster_plot_and_spike_counts(t_spikes, spike_counts)
plot_stimulus(stimulus)


### punto 4



###

plt.show()