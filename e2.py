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

    t1 = ProbabilityVector({'Cases rising': 0.85, 'Cases falling': 0.15}) # Estado 'COVID-19 cases rising'
    t2 = ProbabilityVector({'Cases rising': 0.7, 'Cases falling': 0.3}) # Estado 'COVID-19 cases falling'

    # Se obtiene la matriz de transición de estados (P) según las probabilidades de transición
    # para cada uno de los estados escondidos, en este caso 2.
    T = ProbabilityMatrix({'Cases rising': t1, 'Cases falling': t2}) # Matriz de transición de estados

    print('======= [Matriz de transición de estados] =======')
    print(T)
    print(T.df)

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
    hmc = HiddenMarkovChain_Uncover(T, E, pi)

    print('[Hidden Markov Chain]')
    print(hmc)

    print('')
    print('')

    # Se verifica el modelo utilizando una cadena de observaciones definida previamente
    # El modelo debe arrojar la probabilidad de ocurrencia de dichas observaciones en base
    # al diagrama inicial
    cadena_observacion = ['Go to the park', 'Buy a mask', 'Get infected', 'Get infected', 'Die']
    # Se obtiene la probabilidad o score asociada a la ocurrencia de dichos estados en el modelo
    probabilidad_cadena_observacion = hmc.score(cadena_observacion)
    
    print('======= [Prueba de probabilidad mediante una cadena de observaciones] =======')
    print('- Cadena de observaciones: ' +str(cadena_observacion))
    print('- Probabilidad de ocurrencia: ' +str(probabilidad_cadena_observacion))

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

    print('')
    print('')


    # Se obtienen simulaciones, es decir, listas de las observaciones en base a la
    # secuencia de estados ocultos generados
    secuencia_observada, secuencia_estados_ocultos = hmc.run(4) # Se toma una cantidad n+1 de muestras
    secuencia_descubierta = hmc.uncover(secuencia_observada)

    print('======= [Datos de la simulación] =======')
    print('- Secuencia de observaciones: ' + str(secuencia_observada))
    print('- Secuencia de estados ocultos: ' +str(secuencia_estados_ocultos))
    print('- Secuencia de estados ocultos descubiertos por el modelo: ' +str(secuencia_descubierta))

    print('')
    print('')

    # Se obtiene la tabla de posibles secuencias de estados ocultos a partir de
    # una cadena de observaciones determinada
     
    lista_posibles_estados = {'Cases rising', 'Cases falling'}
    largo_cadena = 5  # any int > 0

    df = get_simulation_score(lista_posibles_estados, largo_cadena, secuencia_observada)

    print('======= [Tabla de ranking de posibles secuencias de estados] =======')
    print('- Secuencia de observaciones: '+str(secuencia_observada))
    print('- Posibles secuencias de estados ocultos:')
    print(df)


    print('')
    print('')


    # Entrenamiento
    observaciones = ['Buy a mask', 'Go to the park', 'Die', 'Get infected', 'Buy a mask', 'Buy a mask']
    estados = ['Cases rising', 'Cases falling']
    observables = ['Die', 'Get infected', 'Buy a mask', 'Go to the park']

    hml = HiddenMarkovLayer(T, E, pi)
    hml.initialize(estados, observables)
    hmm = HiddenMarkovModel(hml)

    print('[Entrenamiento del modelo]')

    try:
        hmm.train(observaciones, 20)
    except:
        pass

    EJECUCIONES = 100000
    T = 5

    cadenas = EJECUCIONES * [0]
    for i in range(len(cadenas)):
        chain = hmm.layer.run(T)[0]
        cadenas[i] = '-'.join(chain)

    df = pd.DataFrame(pd.Series(cadenas).value_counts(), columns=['counts']).reset_index().rename(columns={'index': 'chain'})
    df = pd.merge(df, df['chain'].str.split('-', expand=True), left_index=True, right_index=True)

    s = []
    for i in range(T + 1):
        s.append(df.apply(lambda x: x[i] == observaciones[i], axis=1))

    df['matched'] = pd.concat(s, axis=1).sum(axis=1)
    df['counts'] = df['counts'] / EJECUCIONES * 100
    df = df.drop(columns=['chain'])
    df.head(30)

    print('- Observaciones: ' +str(observaciones))

    print('- Dataframe obtenido')
    print(df)
    


