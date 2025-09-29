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

def plot_raster_plot_and_spike_density(t_spikes, spike_density, fs):
    fig, axs = plt.subplots(2)
    t_samples = np.arange(len(spike_density)) / fs
    axs[0].plot(t_samples, spike_density)

    axs[1].eventplot(t_spikes)

neurons = np.loadtxt("dat/spikes.dat", dtype=int)
stimulus = np.loadtxt("dat/stimulus.dat", dtype=float)

def punto_1(neurons, stimulus):
    ISI = np.concatenate([(np.diff(np.where(neuron == 1)).flatten()) for neuron in neurons])*0.1
    plot_ISI_histogram(ISI=ISI)

    ISI_avg = np.mean(ISI)
    ISI_std = np.std(ISI)
    CV = ISI_std/ISI_avg

    print("ISI average: ", ISI_avg)
    print("ISI std: ", ISI_std)
    print("CV: ", CV)

def punto_2(neurons, stimulus):
    spike_count = [np.count_nonzero(neuron) for neuron in neurons]

    spike_count_avg = np.mean(spike_count)
    spike_count_std = np.std(spike_count)
    FF = spike_count_std**2/spike_count_avg

    print("Spike count average: ", spike_count_avg)
    print("Spike count std: ", spike_count_std)
    print("FF: ", FF)

    plot_spike_count_histogram(spike_count, n_bins=10)
    plot_spike_count_histogram(spike_count, n_bins=20)
    plot_spike_count_histogram(spike_count, n_bins=30)

def punto_3_y_4(neurons, stimulus):
    # punto 3
    def calc_spike_density(neurons, binlen, step, fs):
        '''
            - neurons: 
                - rows: trials
                - column: presence or absence of spike (1 for presence, 0 for absence)
            - fs: sampling freq in kHz
        '''
        n_samples = len(neurons[0])
        n_trials = len(neurons)
        windows_start = range(0, n_samples, step)

        spike_counts = [np.count_nonzero(neurons[:, window_start:window_start+binlen]) for window_start in windows_start]
        
        return np.repeat(np.array([
            spike_count/n_trials/np.minimum(binlen, n_samples-i)*fs 
            for i, spike_count in enumerate(spike_counts)
        ]), step)[:n_samples]

    fs = 10 #kHz
    binlen = 200
    step = 1
    spike_density = calc_spike_density(neurons, binlen, step, fs)

    t_spikes = [np.where(neuron==1)[0]/fs for neuron in neurons]   #ms

    plot_raster_plot_and_spike_density(t_spikes, spike_density, fs)

    # punto 4

    def calc_STA(s, tau, t_spikes):
        if not tau % 1000:
            print(tau)
        padded_s = np.pad(s, (0, len(s)), 'constant', constant_values=(0,0))
        suma = np.sum([padded_s[t_spike-tau] for t_spike in t_spikes])
        return suma / len(t_spikes)
    
    t_spikes = np.concatenate([np.where(neuron==1)[0] for neuron in neurons[0:5]])
    t_stimulus = stimulus[:,0]
    value_stimulus = stimulus[:,1]
    STA = np.array([calc_STA(value_stimulus, tau, t_spikes) for tau in range(len(stimulus))])

    plt.figure()
    plt.plot(t_stimulus, STA)

    var_stim = np.var(stimulus)

    # estimo r y calculo error cuadratico medio
    mses = []
    for binlen in [10,20,100,200]:
        density = calc_spike_density(neurons, binlen, 1, fs)
        r_mean = np.mean(density)
        D = 20*STA * r_mean / var_stim
        r_est = (np.mean(spike_density) + np.convolve(D, value_stimulus))[:10000]
        mses.append(np.square(density-r_est).mean())
        plt.figure()
        plt.plot(density)
        plt.plot(r_est)

    print(mses)
    plt.figure()
    plt.plot(mses)





if __name__ == "__main__":
    # punto_1(neurons, stimulus)
    # punto_2(neurons, stimulus)
    punto_3_y_4(neurons, stimulus)


###

plt.show()