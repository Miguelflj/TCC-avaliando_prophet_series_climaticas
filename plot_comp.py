
import matplotlib.pyplot as plt
import numpy as np

app = ['Prophet sem regressores', 'Prophet com regressores', 'LSTM']
estacoes = ['Belo Horizonte', 'Salinas','Caratinga','Irati']

ind = np.arange(len(estacoes))
width = 0.2



bar_p_sreg = plt.bar(ind, [9.425, 12.988, 7.141, 7.189], width, color='r',label='Prophet s-reg')
bar_p_creg = plt.bar(ind+width, [7.974, 11.976, 6.598, 6.607], width, color='b',label='Prophet c-reg')
bar_lstm = plt.bar(ind+(2*width), [7.611,10.069,7.222,6.548], width, color='g',label='LSTM')

location = ind+width/2
labels = estacoes
plt.xticks(location, labels)
plt.legend()
plt.title("")
plt.show()
