# =====================================================================
# Tarea 1 - Inteligencia Artificial (01-2021)
# Sebastián Ignacio Toro Severino (sebastian.toro1@mail.udp.cl)
# Ejercicio 1
# =====================================================================

import os
import pandas as pd
import bnlearn as bn

# Función que permite cargar el dataset a utilizar desde un archivo .csv
# además de especificar las columnas y filas para realizar la extracción de datos
def cargar_dataset():
    try:
        ruta_carpeta_datasets = './datasets' # Carpeta por defecto para almacenar datsets

        print('[Carga de archivo] Recuerde que el archivo debe encontrarse dentro de la carpeta "datasets".')
        nombre_archivo_dataset = input('Ingrese el nombre del archivo (con su extensión) del dataset a utilizar: ')

        # Se genera la ruta en base a la ruta de la carpeta de datasets y el nombre + extensión del archivo
        ruta_archivo = os.path.normpath(os.path.join(ruta_carpeta_datasets, nombre_archivo_dataset))

        # Se vuelve a solicitar el nombre del archivo en caso de que no se encuentre
        while os.path.exists(ruta_archivo) is not True:
            print('[Error] No se ha encontrado el archivo. Asegúrese que se encuentra dentro de la carpeta "datasets".')
            nombre_archivo_dataset = input('Ingrese el nombre del archivo (con su extensión) del dataset a utilizar: ')

            ruta_archivo = os.path.normpath(os.path.join(ruta_carpeta_datasets, nombre_archivo_dataset))
        
        # Se seleccionan las columnas del archivo
        columnas = input('Ingrese las columnas a utilizar separadas por "-" (ej: 1-4-10-13-23): ')
        # Se separan las columnas ingresadas y se convierten en un arreglo numérico para utilizarlo al leer el archivo
        lista_columnas = [int(num_columna) for num_columna in columnas.replace(' ', '').split('-')]

        # Se obtiene la cantidad de filas a filtrar
        numero_filas = input('Ingrese el número de filas a utilizar: ')
        while numero_filas.isnumeric() is not True:
            print('[Error] El número de filas debe ser un valor numérico.')
            numero_filas = input('Ingrese el número de filas a utilizar: ') 
        
        # Mediante la librería Pandas se importa el archivo csv según 
        # las resitricciones de columnas y filas para almcanear todo como un dataframe
        df = pd.read_csv(ruta_archivo, usecols=lista_columnas, nrows=int(numero_filas))

        print('---------------------------------------- DATAFRAME ASOCIADO ----------------------------------------')
        print('- Archivo seleccionado: '+ str(ruta_archivo))
        print('- Lista de columnas seleccionadas: '+ str(lista_columnas))
        print('- Número de filas seleccionadas: '+ str(numero_filas)) 
        print('* Dataframe:')
        print(df)
        print('----------------------------------------------------------------------------------------------------')

        return df
    
    except Exception as error:
        print('Ha ocurrido el siguiente error al realizar la carga del dataset:')
        print(error)
        return None

# Función para obtención de DAG asociado
def aprendizaje_estructura(df):
    try:
        # Mediante el método structure_learning de la librería bnlearn se obtiene el DAG
        # correspondiente a la relación entre los componentes de la red para retornarlo
        modelo = bn.structure_learning.fit(df)
        print('=================================== MODELO DEL APRENDIZAJE DE ESTRUCTURA')
        print(modelo)
        

        grafico_modelo(modelo)

        return modelo
    
    except Exception as error:
        print('Ha ocurrido el siguiente error al realizar el aprendizaje de estructura:')
        print(error)
        return None

# Función para mejora y obtención de probabilidades en base al DAG obtenido
def aprendizaje_parametros(modelo, df):
    try:
        # Mediante el método parameter_learning de la librería bnlearn se obtiene un modelo actualizado
        # en base al modelo inicial y al dataframe entregado para obtener las probabilidades de los componentes
        modelo = bn.parameter_learning.fit(modelo, df)
        print('=================================== MODELO ACTUALIZADO DEL APRENDIZAJE DE PARÁMETROS')
        print(modelo)

        grafico_modelo(modelo)

        return modelo
    
    except Exception as error:
        print('Ha ocurrido el siguiente error al realizar el aprendizaje de parámetros:')
        print(error)
        return None

# Función para imprimir la distribución de probabilidad condicionada (CPD) del modelo
def imprimir_cpd(modelo):
    bn.print_CPD(modelo)

# Función para graficar el modelo obtenido
def grafico_modelo(modelo):
    bn.plot(modelo)

# Función principal para ejecutar los distintos pasos del proceso
def menu():
    os.system('clear' if os.name == 'posix' else 'cls')
    df = None
    modelo = None

    while True:
        print('')
        print('[1] Cargar dataset')
        print('[2] Aprendizaje de estructura')
        print('[3] Aprendizaje de parámetros')
        print('[4] Imprimir CPD (distribución de probabilidad condicionada)')
        print('[5] Salir')
        print('')
        op = input('> Seleccione una opción: ')
        while op.isnumeric() is not True:
            op = input('> Seleccione una opción: ')

        op = int(op) 
        os.system('clear' if os.name == 'posix' else 'cls')

        if op == 2 and df is None:
            print('[Error] Primero debe cargar un dataset (op: 1).')
            print('')
            continue
        
        elif op == 3 and df is None and modelo is None:
            print('[Error] Primero debe cargar un dataset y realizar el aprendizaje de estructura (op: 1 y 2).')
            print('')
            continue

        elif op == 4 and df is None and modelo is None:
            print('[Error] Primero debe cargar un dataset, realizar aprendizaje de estructura y parámetros (op: 1, 2, 3).')
            print('')
            continue

        if op == 1:
            df = cargar_dataset()
        
        elif op == 2:
            modelo = aprendizaje_estructura(df)
        
        elif op == 3:
            modelo = aprendizaje_parametros(modelo, df)

        elif op == 4:
            imprimir_cpd(modelo)

        elif op == 5:
            break


if __name__ == '__main__':
    menu()