
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


#todo(position: "inline")[es renewal?]

= Tasa de disparo dependiente del tiempo

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




#todo(position: "inline")[grafico D]

#todo(position: "inline")[codigo de calculo de D]


= Anexo
El repostorio de código utilizado para simular y graficar se encuentra en: #box[#link( "https://github.com/malustewart/redes-neuronales-tp3" )]
