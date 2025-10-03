
#import "@preview/basic-report:0.3.1": *
#import "@preview/dashy-todo:0.1.2": todo

#set math.equation(numbering: "(1)")

#show: it => basic-report(
  doc-category: "Redes neuronales",
  doc-title: "Trabajo práctico 3: \nEstadística de trenes de disparo",
  author: "María Luz Stewart Harris",
  affiliation: "Instituto Balseiro",
  language: "es",
  compact-mode: true,
  heading-font: "Vollkorn",
  heading-color: black,
  it
)

= Distribución de disparos y coeficiente de variabilidad
La #ref(<fig:hist_isi>) muestra un histograma de los intervalos entre disparos (ISI).

#figure(
  image("figs/histogram_isi.png"),
  caption: "Histograma del intervalo entre disparos (ISI)."
)<fig:hist_isi>

El ISI fue calculado con el siguiente código:
```python
def calc_ISI(neurons, T_ms):
    ISI = [(np.diff(np.where(neuron == 1)).flatten()) for neuron in neurons]
    ISI = np.concatenate(ISI)
    ISI = ISI*T_ms
    return ISI
```

A partir de la medición de actividad se calculó el coeficiente de variabilidad (CV): 

$ "CV" = sigma_("ISI") / #overline[ISI] = 5.632/8.569 approx 0.657 $

= Cantidad de disparos y factor de Fano

La #ref(<fig:hist_N>) muestra un histograma de la cantidad de disparos por muestra de 1s ($N$).

#figure(
  image("figs/histogram_N.png"),
  caption: "Histograma de cantidad de disparos por muestra."
)<fig:hist_N>

Se calculó el factor de Fano (FF):

$ "FF" = sigma_("N")^2 / #overline[N] = (13.53)^2/117.01 approx 1.566 $


Dado que $"FF"^2 != "CV"$, no es un proceso de renewal. 

= Tasa de disparo dependiente del tiempo <sec:r_t>

Se calculó la densidad de disparos $r(t)$ utilizando un ancho de bin de 100 muestras (10ms).

$ r(t) = 1/(Delta t) (n_(K)(t;t+Delta t))/K $


Donde
 - $Delta t=10$ms es el ancho de bin
 - $K=128$ es la cantidad de neuronas
 - $n_K (t; t + Delta t)$ es la cantidad de disparos ocurridos entre $t$ y $t+Delta t$ en todas las neuronas

La #ref(<fig:r_t>) muestra el resultado del cálculo de $r(t)$ y el raster plot de todos los disparos.

#figure(
  image("figs/r_t.png"),
  caption: "Estimación de la frecuencia de disparos (longitud de bin: 10ms) y raster plot de todos los disparos."
)<fig:r_t>

= Filtro asociado a la neurona

Se busca un filtro para estimar $r(t)$:
$ r_"est"(t) = r_0 + integral_0^infinity D(tau)s(t-tau) dif tau $

== Cálculo $r_0$

$r_0$ se obtuvo de acuerdo a la siguiente ecuación:$ r_0 = 1/N_T sum_(i=0)^N_T r_i $

Calculado a partir del $r(t)$ obtenido en la #ref(<sec:r_t>), se obtiene:

$ r_0 = 116.2 "Hz" $


== Cálculo $D(tau)$

$D(tau)$ se obtuvo de acuerdo a la siguiente ecuación:

$ D(tau) = 1/sigma_s^2 angle.l r angle.r "STA"(tau) $

donde
$ "STA"(tau) = 1/N_("spikes")sum_({n_("spikes")}) s(t_("spikes")-tau) $

El siguiente código fue utilizado para calcular $D(tau)$:

```python
def calc_STA(s, tau, t_spikes):
    suma = np.sum([s[t_spike-tau] for t_spike in t_spikes if t_spike - tau >= 0])
    return suma / len(t_spikes)

STA = np.array([calc_STA(value_stimulus, tau, t_spikes) for tau in range(len(stimulus))])

D = STA*r_mean/var_stimulus
```

La #ref(<fig:D>) muestra la función $D(tau)$ resultante.

#figure(
  image("figs/D.png"),
  caption: [Filtro lineal D que predice $r(t)-angle.l r angle.r$ en función del estímulo.]
)<fig:D>

= Anexo
El repostorio de código utilizado para simular y graficar se encuentra en: #box[#link( "https://github.com/malustewart/redes-neuronales-tp3" )]
