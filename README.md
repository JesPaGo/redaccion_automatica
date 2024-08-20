# Agente inteligente recuperación sensórica IoT
## Descripción
Este proyecto ha sido desarrollado como parte de un reto propuesto por la empresa en el Matchmaking organizado por la Universidad de Granada. Se ha liberado una parte del código totalmente funcional como parte del trabajo de fin de máster para mostrar su implementación y su modo de uso.

Consiste en la creación de un modelo de Lenguaje de Aprendizaje Automático utilizable como agente inteligente para consultar y seleccionar materiales IoT en función de sus especificaciones y el proveedor. 

 ---
## Instalación
Para poder ejecutar el código, es necesario desplegar un entorno virtual con las librerías requeridas.

La instalación del entorno virtual se realiza a través de Anaconda usando el archivo [```environment.yml```](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/environment.yml).
En la consola del sistema, ejecutamos el siguiente comando:
```
conda env create -f environment.yml
```
---
## Modo de uso
Como ejemplo de uso seguiremos el flujo de trabajo especificado en [```demo.ipynb```](). 

1. Carga de todas las librerías necesarias para ejecutar el agente.

Es fundamental ajustar el ```path_root``` para que apunte hacia el origen de la carpeta.
```
path_root = r"C:/Users/jgpg000.edu/Desktop/Agente inteligente recuperación sensórica IoT/"

# ruta clases personalizadas
import sys
path_classes = f"{path_root}classes"
sys.path.append(path_classes)

## LIBRERIAS
import pandas as pd
from class_data import *
from fun_models import *
```

2. Activación de la clave de OpenAI en el sistema.

Añadimos la clave al ```.env```.
```
#Incluimos la clave de la API de OpenAI 
OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
```
De esta forma queda almacenada como una variable local del proyecto. Podemos acceder a ella con:
```
api_key=os.environ.get("OPENAI_API_KEY")
```

La opción recomendada por OpenAI es implementar la variable de forma global. Los pasos a seguir vienen explicados en la propia [guía de la empresa](https://platform.openai.com/docs/quickstart).

3. Carga de la base de datos

En el ejemplo se cargan por defecto los datos de los proyectos de Las Rozas, Oviedo, Elche y Mérida.
Los datos se almacenan en la ruta `{path_root}datasets/database_items/`. 

La primera vez que se crea la base de datos también se almacena internamente en formato **json**.
Por ello, es necesario identificar la ruta donde se almacena esta información que por defecto es:
`{path_root}datasets/database_items/database_json/`. 

Al guardar la información en este formato accedemos más rápido a la misma y se evita tener que llamar a un LLM (de OpenAI) cada vez que se carguen los documentos.
```

path_excel = f"{path_root}datasets/database_items/item_list.xlsx" 
path_saved_json = f"{path_root}datasets/database_items/database_json/" 

documentos = load_documents(path_excel, path_saved_json) 
documentos_langchain = load_documents_langchain(documentos) 
```
La base de datos a cargar está especificada en el documento [`item_list.xlsx`]().
Al ejecutar la celda de arriba, debe devolver el siguiente mensaje.
```
100%|██████████| 88/88 [00:00<00:00, 149.17it/s]
Se han cargado 86 documentos.
100%|██████████| 86/86 [00:00<00:00, 3365.59it/s]
Se han generado 954 documentos de Langchain.
De ellos, 86 son el texto y 868 son resúmenes de tablas.
```
Estos documentos deben trocearse en elementos más pequeños y de longitud fija. Se aplica un *text_splitter* de Langchain.
```
text_splitter = RecursiveCharacterTextSplitter( 
    chunk_size=300, 
    chunk_overlap=30, 
    length_function=len, 
    is_separator_regex=False,
) 
# Dividimos cada texto en chunks 
chunks_texto = text_splitter.split_documents(documentos_langchain) 
print(f"Hay {len(chunks_texto)} chunks de texto.") 
```
Nos devuelve el mensaje:
```
Hay 7218 chunks de texto.
```
Esta base de datos se puede ampliar siguiendo la metodología descrita en **[Ampliación de la base de datos]()**.

4. Configuración del agente

Hay que definir varios parámetros para configurar el agente:

- *path_saved_vector*: Ruta donde se guardan los vectores. 
- *modelo_llm*: Modelo de OpenAI usado para generar la respuesta. 
- *template_sistema*: Instrucciones para que funcione el modelo. 
```
path_saved_vector = f"{path_root}datasets/database_items/database_vectorial/" 
name_database = "large_esp" 
modelo_llm = "gpt-4o" 

template_sistema = """ Eres un asistente de una empresa encargada de smart cities.
Como preguntas, te van a adjuntar listas de requisitos de equipo para proyecto. 
Tu labor es responder con el nombre de los equipos que encuentres que cumplan los requisitos, así como suministrar la información de estos dispositivos en relación a los requisitos. 
Usa la información añadida en el contexto para responder a la pregunta pero no hagas referencias al mismo. 
Se añade como metadato tanto el proveedor como el nombre a los que pertenece el contexto devuelto. 
Responde únicamente con la especificación que se pide. Si no aparece di que no lo sabes. 
Pregunta: {question} 
Metadato: {metadata} 
Contexto: {contexts} 
La respuesta debe tener el siguiente formato: 
- Nombre del equipo y proveedor. 
- Especificaciones que se han preguntado y aparecen en el contexto. 
Respuesta: 
""" 
```

Se incluyen tres funciones para una inicialización rápida del agente:
```
vectorstore = create_database(f"{path_saved_vector}{name_database}",
                               chunks_texto, nombre_embedding = "text-embedding-3-large")
retriever = create_retriever(vectorstore)
rag = create_RAG(retriever=retriever,
                plantilla_RAG=template_sistema,
                model=modelo_llm)
```
Al ejecutar el código debe aparecer el siguiente mensaje.
```
Espacio creado correctamente. Contiene 7218 vectores.
Retriever creado correctamente.
RAG creado correctamente.
    Llámalo con <rag_name>.invoke(<query>)
```

5. Interacción con el agente

En este punto ya podemos hacer consultas a nuestro agente.
El formato óptimo de una consulta viene detallado en  [Formato de preguntas para el modelo]().

Se utilizará la siguiente pregunta de ejemplo en esta demostración.
```
pregunta_ejemplo = """Necesitamos un sensor de medición de ruido con las siguientes características: 
1. Sensor con precisión clase 1 según IEC 61672-1: 
o Detector: Nivel de presión sonora continuo equivalente y nivel de presión sonora con ponderación temporal rápida (Fast) y lenta (Slow). 
o Ponderación Frecuencial: A y C. 
o Funciones Acústicas Medidas: Niveles equivalentes con ponderación frecuencial A y C con tiempo programable entre 1s y 60min: LAeqT y LCeqT. Niveles máximos con ponderación temporal rápida y slow sobre un tiempo programable entre 1s y 60min y ponderación frecuencial A: LAFmaxT y LASmaxT. 
o Resolución 0,1 dB. 
o Precisión según IEC 61672-1: clase 1. 
o Margen de medición sin escalas: de 28 a 120 dBA. 
o Margen de linealidad a 1kHz : de 35 a 120 dBA. 
2. Protección contra agentes externos con kit de exterior: viento, lluvia, pájaros. Mantiene clase 1. Protección IP65. 
3. Capacidad de integración a una plataforma de monitorización de ruido, de código abierto o Propietarias. 
4. Dimensiones reducidas y fácil de instalar en farolas, luminarias, marquesinas, MUPIs, OPIs, vallas y postes publicitarios. 
5. Alimentación de energía, a través la red eléctrica, baterías externas, paneles solares o POE (Power Over Ethernet). 
6. Medición continua 24 horas/7 días a la semana. 
7. Red sin límite en el número de sensores. 
8. Comunicación por Ethernet RJ45, Wi-Fi, Modem. 
9. Configuración remota del sensor, sin necesidad de tener dirigirse hasta el lugar. """
```
Llamamos al modelo con:
```
respuesta = rag.invoke(pregunta_ejemplo)
print(respuesta['answer'])
```
Y devuelve:
```
'- Nombre del equipo y proveedor: CESVA Clase 1 modelo TA120, CESVA.\n- Especificaciones que se han preguntado y aparecen en el contexto:\n  - Sensor con precisión clase 1 según IEC 61672-1.\n  - Protección contra agentes externos con kit de exterior: viento, lluvia, pájaros. Mantiene clase 1. Protección IP65.\n  - Completamente integrable en diferentes plataformas: Noise-Platform (CESVA), de código abierto como Sentilo.\n  - Margen de medición sin escalas: de 28 a 120 dBA.\n  - Margen de linealidad a 1kHz: de 35 a 120 dBA.\n  - Red sin límite en el número de sensores.\n  - Comunicación por Ethernet (RJ45), Wi-Fi, Módem.\n  - Configuración remota del sensor.'
```
Es importante especificar dentro de la respuesta *answer* pues una llamada al agente crea un diccionario con cuatro entradas: *contexts, question, metadata, answer*.

**NOTA:** Por defecto, el agente no incluye un registro de la conversación.
Únicamente es capaz de contestar al último mensaje enviado.

6. Interfaz gráfica

Se puede implementar una interfaz gráfica de manera sencilla usando la librería *Gradio*.
```
import gradio as gr
def predict(message, history):
    response = rag.invoke(message)
    return response['answer']

gr.ChatInterface(predict).launch()

```
Tras ejecutar el comando se abre una interfaz tanto en el entorno de programación como en el URL local: **http://127.0.0.1:7860**.
Esta opción es análoga a llamar al modelo con `.invoke`.

![Interfaz por defecto de Gradio](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/figs/interfaz_gradio.png)

---
## Modelos de agente

La estructura de los cuadernos sobre modelos es la siguiente:
1. Librerías y clases personalizadas.
2. Integración de la base de datos (a partir del excel).
3. Especificaciones del modelo RAG: indicación de rutas, templates y modelos. 
4. Construcción del modelo: vectorstore, retriever, (reranker), rag. 
5. Ejemplo de respuesta y funcionamiento.
6. Interfaz gráfica.

- [ ] [Agentes básicos](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/agents/agente-basico.md)
- [ ] [Agentes basados en moda](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/agents/agente-moda.md)
- [ ] [Agentes con reranker](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/agents/agente-reranker.md)
- [ ] [Agentes de conjunto (ensemble)](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/agents/agente-ensemble.md)
---
## Funcionalidades
- [ ] [Creación de un Excel de productos](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/crear-excel.md)
- [ ] [Lista de equipos](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/lista-equipo.md)
- [ ] [Ampliación de la base de datos](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/ampliar-basedatos.md)
- [ ] [Formato de preguntas para el modelo](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/formato-preguntas.md)
- [ ] [Creación de un conjunto de datos de test](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/crear-datos-test.md)
- [ ] [Validación de un conjunto de datos](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/validacion-datos.md)
- [ ] [Lista de métricas](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/lista-metricas.md)
- [ ] (WIP) [Implementar historial de conversación](https://gitlab.innovasur.es/ia/agente-inteligente-recuperacion-sensorica-iot/docs/historial-conversacion.md)
---
## Soporte
Ante cualquier duda, puede contactar conmigo a través del correo:
```
jgpg000.edu@innovasur.com
```
---
## Hoja de ruta
- Implementación de un historial de conversación
- Redirigir los enlaces de los Markdown