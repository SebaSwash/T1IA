# =====================================================================
# Tarea 1 - Inteligencia Artificial (01-2021)
# Sebastián Ignacio Toro Severino (sebastian.toro1@mail.udp.cl)
# Ejercicio 2
# =====================================================================
import os
import numpy as np
import pandas as pd
from itertools import product
from functools import reduce
from matplotlib import pyplot as plt
from HmmObjects import *

# Función para poder obtener mediante un ranking (con puntaje) la similitud entre
# la secuencia de observaciones obtenida en la simulación y las posibles secuencias
# de estados ocultos que pueden formarse a partir de tales observaciones
def get_simulation_score(all_possible_states, chain_length, observed_sequence):
    all_states_chains = list(product(*(all_possible_states,) * chain_length))

    df = pd.DataFrame(all_states_chains)
    dfp = pd.DataFrame()

    for i in range(chain_length):
        dfp['p' + str(i)] = df.apply(lambda x: 
            hmc.E.df.loc[x[i], observed_sequence[i]], axis=1)
    
    scores = dfp.sum(axis=1).sort_values(ascending=False)
    df = df.iloc[scores.index]
    df['score'] = scores
    df.head(10).reset_index()

    return df

if __name__ == '__main__':
    os.system('clear' if os.name == 'posix' else 'cls')

    p1 = ProbabilityVector({'Cases rising': 0.85, 'Cases falling': 0.15}) # Estado 'COVID-19 cases rising'
    p2 = ProbabilityVector({'Cases rising': 0.7, 'Cases falling': 0.3}) # Estado 'COVID-19 cases falling'

    # Se obtiene la matriz de transición de estados (P) según las probabilidades de transición
    # para cada uno de los estados escondidos, en este caso 2.
    P = ProbabilityMatrix({'Cases rising': p1, 'Cases falling': p2}) # Matriz de transición de estados

    print('======= [Matriz de transición de estados] =======')
    print(P)
    print(P.df)

    print('')
    print('')

    e1 = ProbabilityVector({'Die': 0.35, 'Get infected': 0.4, 'Buy a mask': 0.25, 'Go to the park': 0.0}) # Observaciones para estado 'COVID-19 cases rising'
    e2 = ProbabilityVector({'Die': 0.2, 'Get infected': 0.3, 'Buy a mask': 0.1, 'Go to the park': 0.4}) # Observaciones para estado 'COVID-19 cases falling'
    
    # Se obtiene la matriz de emisión según las observaciones y estados escondidos del diagrama.
    E = ProbabilityMatrix({'Cases rising': e1, 'Cases falling': e2})

    print('======= [Matriz de emisión] =======')
    print(E)
    print(E.df)

    print('')
    print('')

    # Probabilidades iniciales de estados (pi)
    pi = ProbabilityVector({'Cases rising': 0.7, 'Cases falling': 0.3})

    print('======= [Distribución inicial de probabilidades (pi)] =======')
    print(pi.df)

    print('')
    print('')

    # Se genera la HMC (hidden Markov chain) en base a las matrices de transición, emisión y 
    # probabilidades iniciales de estados
    hmc = HiddenMarkovChain_Uncover(P, E, pi)

    print('[Hidden Markov Chain]')
    print(hmc)

    print('')
    print('')

    # Se comprueba que la implementación sea correcta.
    # Para ello, la suma de todas las probabilidades para
    # todos las posibles cadenas de observación deben sumar 1.
    lista_posibles_observaciones = {'Die', 'Get infected', 'Buy a mask', 'Go to the park'}
    largo_cadena = 4  # any int > 0
    lista_cadenas_observaciones = list(product(*(lista_posibles_observaciones,) * largo_cadena))
    lista_posibles_probabilidades = list(map(lambda obs: hmc.score(obs), lista_cadenas_observaciones))

    print('======= [Validación de implementación] =======')
    print('- Suma de la lista de posibles probabilidades: ' + str(sum(lista_posibles_probabilidades)))


    # Se obtienen simulaciones, es decir, listas de las observaciones en base a la secuencia de estados ocultos
    #observed_sequence, latent_sequence = hmc.run(5) # Se toma una cantidad n+1 de muestras
    #uncovered_sequence = hmc.uncover(observed_sequence)

    #print('[Datos de la simulación]')
    #print('- Secuencia de observaciones: ' + str(observed_sequence))
    #print('- Secuencia de estados ocultos: ' +str(latent_sequence))

    #print('')
    #print('')

    #print('- Secuencia de estados descubierta: ' +str(uncovered_sequence))

    #all_possible_states = {'Cases rising', 'Cases falling'}
    #chain_length = 6  # any int > 0

    #df = get_simulation_score(all_possible_states, chain_length, observed_sequence)
    #print(df)


