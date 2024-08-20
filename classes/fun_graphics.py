path_root = r"C:/Users/jgpg000.edu/Desktop/LIBERAR TFM/"
## LIBRERIAS
import pandas as pd
from class_dataset import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from adjustText import adjust_text

def grafico_comparacion(experimento1: str, experimento2: str, position_legend: str = "upper left",
                    ruta_grafico: str = f"{path_root}testing/figures_test/",
                    ruta_dataset_csv: str = f"{path_root}testing/responses_test/csv/"): 
    '''
    Compara y guarda una figura con los valores medios de las métricas de dos experimentos.
    Parámetros:
    - experimento1 (str): nombre del primer experimento.
    - experimento2 (str): nombre del segundo experimento.
    - position_legend (str): posición de la leyenda en la figura.
    - ruta_grafico (str): ruta donde se guardará la figura.
    - ruta_dataset_csv (str): ruta donde se encuentran los archivos CSV con los resultados de los experimentos.
    '''
    ## Cargamos los modelos
    qa_exp1 = QADataset(f"{ruta_dataset_csv}{experimento1}.csv")
    qa_exp2 = QADataset(f"{ruta_dataset_csv}{experimento2}.csv")
    metrica_exp1 = qa_exp1.show_metrics().mean(axis=0)
    metrica_exp2 = qa_exp2.show_metrics().mean(axis=0)
    df = pd.DataFrame([metrica_exp1, metrica_exp2], index=["Baseline", "Filtro Metadato"]).T
    df.drop(["answer_relevancy", "especificaciones_correctas", "context_recall", "context_relevancy"], inplace=True)

    metricas = df.index.tolist()
    baseline = df['Baseline'].tolist()
    filtro_metadato = df['Filtro Metadato'].tolist()

    # Crear el gráfico
    x = np.arange(len(metricas))  # etiquetas de las métricas
    width = 0.35  # el ancho de las barras

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline, width, label=experimento1)
    rects2 = ax.bar(x + width/2, filtro_metadato, width, label=experimento2)

    # Agregar algunas etiquetas
    ax.set_ylabel('Valores')
    #ax.set_title(f'Comparación de métricas entre {experimento1} y {experimento2}')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, rotation=45, ha="right")
    ax.legend(loc=position_legend)

    # Función para agregar etiquetas encima de las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig(f"{ruta_grafico}comparacion_{experimento1}_{experimento2}.png")
    plt.show()

def grafico_comparacion3(experimento1: str, experimento2: str, experimento3: str,
                    ruta_grafico: str = f"{path_root}testing/figures_test/",
                    ruta_dataset_csv: str = f"{path_root}testing/responses_test/csv/"): 
    '''
    Compara y guarda una figura con los valores medios de las métricas de tres experimentos.
    Parámetros:
    - experimento1 (str): nombre del primer experimento.
    - experimento2 (str): nombre del segundo experimento.
    - experimento3 (str): nombre del tercer experimento.
    - ruta_grafico (str): ruta donde se guardará la figura.
    - ruta_dataset_csv (str): ruta donde se encuentran los archivos CSV con los resultados de los experimentos.
    '''
    ## Cargamos los modelos
    qa_exp1 = QADataset(f"{ruta_dataset_csv}{experimento1}.csv")
    qa_exp2 = QADataset(f"{ruta_dataset_csv}{experimento2}.csv")
    qa_exp3 = QADataset(f"{ruta_dataset_csv}{experimento3}.csv")
    metrica_exp1 = qa_exp1.show_metrics().mean(axis=0)
    metrica_exp2 = qa_exp2.show_metrics().mean(axis=0)
    metrica_exp3 = qa_exp3.show_metrics().mean(axis=0)
    df = pd.DataFrame([metrica_exp1, metrica_exp2, metrica_exp3], index=["EXP1", "EXP2", "EXP3"]).T
    df.drop(["answer_relevancy", "especificaciones_correctas", "context_recall", "context_relevancy"], inplace=True)

    metricas = df.index.tolist()
    exp1 = df['EXP1'].tolist()
    exp2 = df['EXP2'].tolist()
    exp3 = df['EXP3'].tolist()

    # Crear el gráfico
    x = np.arange(len(metricas))  # etiquetas de las métricas
    width = 0.25  # el ancho de las barras

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, exp1, width, label=experimento1)
    rects2 = ax.bar(x, exp2, width, label=experimento2)
    rects3 = ax.bar(x + width, exp3, width, label=experimento3)

    # Agregar algunas etiquetas
    ax.set_ylabel('Valores')
    #ax.set_title(f'Comparación de métricas entre {experimento1}, {experimento2} y {experimento3}')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, rotation=45, ha="right")
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Función para agregar etiquetas encima de las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.savefig(f"{ruta_grafico}/comparacion3_{experimento1}_{experimento2}_{experimento3}.png")
    plt.show()

def grafico_comparacion4(experimento1: str, experimento2: str, experimento3: str, experimento4: str,
                    ruta_grafico: str = f"{path_root}testing/figures_test/",
                    ruta_dataset_csv: str = f"{path_root}testing/responses_test/csv/"):
    '''
    Compara y guarda una figura con los valores medios de las métricas de cuatro experimentos.
    Parámetros:
    - experimento1 (str): nombre del primer experimento.
    - experimento2 (str): nombre del segundo experimento.
    - experimento3 (str): nombre del tercer experimento.
    - experimento4 (str): nombre del cuarto experimento.
    - ruta_grafico (str): ruta donde se guardará la figura.
    - ruta_dataset_csv (str): ruta donde se encuentran los archivos CSV con los resultados de los experimentos.
    '''
   ## Cargamos los modelos
    qa_exp1 = QADataset(f"{ruta_dataset_csv}{experimento1}.csv")
    qa_exp2 = QADataset(f"{ruta_dataset_csv}{experimento2}.csv")
    qa_exp3 = QADataset(f"{ruta_dataset_csv}{experimento3}.csv")
    qa_exp4 = QADataset(f"{ruta_dataset_csv}{experimento4}.csv")
    metrica_exp1 = qa_exp1.show_metrics().mean(axis=0)
    metrica_exp2 = qa_exp2.show_metrics().mean(axis=0)
    metrica_exp3 = qa_exp3.show_metrics().mean(axis=0)
    metrica_exp4 = qa_exp4.show_metrics().mean(axis=0)
    df = pd.DataFrame([metrica_exp1, metrica_exp2, metrica_exp3, metrica_exp4], index=["EXP1", "EXP2", "EXP3", "EXP4"]).T
    df.drop(["answer_relevancy", "especificaciones_correctas", "context_recall", "context_relevancy"], inplace=True)

    metricas = df.index.tolist()
    exp1 = df['EXP1'].tolist()
    exp2 = df['EXP2'].tolist()
    exp3 = df['EXP3'].tolist()
    exp4 = df['EXP4'].tolist()

    # Crear el gráfico
    x = np.arange(len(metricas))  # etiquetas de las métricas
    width = 0.24  # el ancho de las barras

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 1.5*width, exp1, width, label=experimento1)
    rects2 = ax.bar(x - 0.5*width, exp2, width, label=experimento2)
    rects3 = ax.bar(x + 0.5*width, exp3, width, label=experimento3)
    rects4 = ax.bar(x + 1.5*width, exp4, width, label=experimento4)

    # Agregar algunas etiquetas
    ax.set_ylabel('Valores')
    #ax.set_title(f'Comparación de métricas entre {experimento1}, {experimento2}, {experimento3} y {experimento4}')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, rotation=45, ha="right")
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Función para agregar etiquetas encima de las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    fig.tight_layout()
    plt.savefig(f"{ruta_grafico}/comparacion4_{experimento1}_{experimento2}_{experimento3}_{experimento4}.png")
    plt.show()

def grafico_comparacion_continuo(patron_nombre_experimento: str, valores_k: list, nombre: str = None,
                        metricas: list = QADataset.columnas_metricas.copy(),
                        ruta_grafico: str = f"{path_root}testing/figures_test/",
                        x_name: str = "Eje X", y_name: str = "Valores de métricas",
                        ruta_dataset_csv: str = f"{path_root}testing/responses_test/csv/"): 
    '''
    Compara y guarda una figura con los valores medios de las métricas de un experimento con distintos valores un mismo modelo.
    Parámetros:
    - patron_nombre_experimento (str): nombre del experimento.
    - valores_k (list): lista con los valores de k.
    - nombre (str): nombre del gráfico.
    - metricas (list): lista con las métricas a comparar.
    - ruta_grafico (str): ruta donde se guardará la figura.
    - x_name (str): nombre del eje X.
    - y_name (str): nombre del eje Y.
    - ruta_dataset_csv (str): ruta donde se encuentran los archivos CSV con los resultados de los experimentos.
    '''
    experimento = patron_nombre_experimento
    valores_metricas = np.zeros((len(metricas), len(valores_k)), dtype=float)

    for col in range(len(valores_k)):
        qa_exp = QADataset(f"{ruta_dataset_csv}{experimento}{valores_k[col]}-Q16.csv")
        qa_metrica = qa_exp.show_metrics().mean(axis=0)
        for fil, metrica in enumerate(metricas):
            valores_metricas[fil, col] = qa_metrica[metrica]

    # creamos un plot que compare las distintas filas de valores_metricas
    fig, ax = plt.subplots(figsize=(13, 8))
    texts = []
    for i in range(len(metricas)):
        ax.plot(valores_k, valores_metricas[i],"o-.", markersize="11", label=metricas[i])
        # añadimos el valor de cada punto
        texts += [ax.text(x, y+0.01, f"{y:.3f}", fontsize=11, ha="center") for x, y in zip(valores_k, valores_metricas[i])]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    ax.set_xlabel(x_name, fontsize=16)
    ax.set_ylabel(y_name, fontsize=16)
    ax.legend(metricas, loc='center left', bbox_to_anchor=(1, 0.5))
    if nombre == None: 
        nombre = patron_nombre_experimento
    #ax.set_title(nombre, fontsize=24)
    plt.savefig(f"{ruta_grafico}/{patron_nombre_experimento}.png")
    plt.show()
   

def comparacion_precio(df, x, y, z,
                ruta_grafico: str = f"{path_root}testing/figures_test/"):
    '''
    Grafica un scatter plot con los precios de los modelos testeados en función de dos especificaciones.
    Parámetros:
    - df (DataFrame): DataFrame con los datos.
    - x (Series): especificación 1.
    - y (Series): especificación 2.
    - z (Series): precios.
    - ruta_grafico (str): ruta donde se guardará la figura.
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(x, y, c=z, s=80, cmap='turbo')
    plt.colorbar().set_label(z.name)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    texts = [plt.text(x[i], y[i], df.nombre_modelo[i], ha='center', va='center') for i in range(len(x)) if re.search(r"_k", df.nombre_modelo[i]) == None]
    texts.append(plt.text(x[17], y[17], df.nombre_modelo[17], ha='center', va='center')) # parent_retriever_reranker_cohere_k_15
    texts.append(plt.text(x[38], y[38], df.nombre_modelo[38], ha='center', va='center')) # pr_cohere_k15_parent3500_childsize_300
    texts.append(plt.text(x[45], y[45], df.nombre_modelo[45], ha='center', va='center')) # pr_cohere_k15_parent3500_child300_filtro      
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    # guardamos la fig
    plt.savefig(f"{ruta_grafico}grafico_precio_especificaciones_{y.name}.png")
    plt.show()