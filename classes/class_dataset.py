import pandas as pd
from datasets import Dataset
from ast import literal_eval
import os
from ragas.evaluation import evaluate
from bert_score import BERTScorer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from class_data import FormatoDato
import matplotlib.pyplot as plt
import tiktoken
from tqdm import tqdm

class QADataset():
    ''' Clase que crea un dataset para validar el RAG.'''
    columnas_dataset = ['id', 'question', 'ground_truth', 'contexts', 'answer', 'equipo']
    columnas_metricas = ['answer_relevancy',
                        'context_relevancy',
                        'context_recall',
                        'answer_similarity',
                        'answer_correctness',
                        "precision_bertscore",
                        "recall_bertscore",
                        "f1_bertscore",
                        'mismo_producto',
                        'especificaciones_correctas',
                        'especificaciones_correctas_norm']
    columnas_contextos = ['contexts_long', 'contexts_proveedores', 'contexts_equipos', 'contexts_modelo']
    columnas = columnas_dataset + columnas_metricas + columnas_contextos

    def name(self):
        return "QADataset"

    def __init__(self, ruta: str, sheet_name: str = None):
        self.ruta = ruta
        # dependiendo de si la ruta termina en .xlsx o .csv se carga de una forma u otra
        if ruta[-4:] == ".csv":
            dataset = self.crear_dataset_desde_csv()
        else:
            dataset = self.crear_dataset_desde_excel(sheet_name)
        # Precargamos igualmente todas las columnas
        for columna in QADataset.columnas:
            if columna in dataset.columns:
                setattr(self, columna, dataset[columna].tolist())
            else:
                setattr(self, columna, ["nan" for x in range(len(dataset))]) # 

    def generar_respuesta_individual(self, rag_model, num_pregunta_dataset: int):
        ''' Elige una pregunta del dataset para generar la respuesta del modelo.
        Args:
            num_pregunta_dataset (int): Número de pregunta del dataset.
            rag_model (RAG): Modelo RAG que se encarga de responder a las preguntas.
        '''
        if self.answer[num_pregunta_dataset] != "nan":
            print("Por favor, responde a la pregunta.")
            seguir = input("Ya se ha generado la respuesta. ¿Quieres sobreescribirla? (y/n)")
            if seguir == "n":
                return
        solution = rag_model.invoke(self.question[num_pregunta_dataset])
        self.contexts[num_pregunta_dataset] = [context.page_content for context in solution["contexts"]]
        self.contexts_long[num_pregunta_dataset] = solution["contexts"]
        # COMPROBAR
        self.contexts_proveedores[num_pregunta_dataset] = [context.metadata["proveedor"] for context in solution["contexts"]]
        self.contexts_equipos[num_pregunta_dataset] = [context.metadata["equipo"] for context in solution["contexts"]]
        self.contexts_modelo[num_pregunta_dataset] = [context.metadata["modelo"] for context in solution["contexts"]]
        self.answer[num_pregunta_dataset] = solution['answer']
        return solution

    def generar_respuestas(self, rag_model):
        ''' Se generan las respuestas a través del modelo RAG.
        Args:
            rag_model (RAG): Modelo RAG que se encarga de responder a las preguntas.
        '''
        if self.answer != ["nan" for x in range(len(self.question))]:
            print("Por favor, responde a la pregunta.")
            seguir = input("Ya se han generado las respuestas. ¿Quieres sobreescribirlas? (y/n)")
            if seguir == "n":
                return
        solutions = [rag_model.invoke(q) for q in self.question]
        self.contexts = [[context.page_content for context in response["contexts"]] for response in solutions]
        self.contexts_long = [response["contexts"] for response in solutions]
        #comprobar
        self.contexts_proveedores = [[context.metadata["proveedor"] for context in response["contexts"]] for response in solutions]
        self.contexts_equipos = [[context.metadata["equipo"] for context in response["contexts"]] for response in solutions]
        self.contexts_modelo = [[context.metadata["modelo"] for context in response["contexts"]] for response in solutions]
        self.answer = [sol['answer'] for sol in solutions]
        print(f"Se han generado las respuestas a las {len(self.question)} preguntas.")

    def show(self):
        dataset = pd.DataFrame({"question": self.question,
                                "answer": self.answer,
                                "contexts": self.contexts,
                                "ground_truth": self.ground_truth
                                })
        return dataset
    
    def to_dataset(self):
        dataset_desde_dict = Dataset.from_dict(self.show())
        return dataset_desde_dict

    def atributos(self):
        return [att for att in self.__dict__.keys()]
    
    def guardar(self, ruta_guardar: str, nombre_archivo: str, sobreescribir: bool = False):
        ''' Guarda el dataset en formato csv.
        Args:
            ruta_guardar (str): Ruta donde se guardará el dataset.
            nombre_archivo (str): Nombre del archivo que se guardará.
        '''
        ruta_total = f"{ruta_guardar}{nombre_archivo}.csv"
        if sobreescribir == False:
            # Comprobamos si el archivo ya existe
            if os.path.exists(ruta_total):
                print("El archivo ya existe.")
                seguir = input("¿Quieres sobreescribirlo? (y/n)")
                if seguir == "n":
                    return
        dataset = self.show()
        # guardamos toda la información
        for columna in QADataset.columnas:
            if columna is None:
                continue
            else:
                dataset[columna] = getattr(self, columna)
        dataset.to_csv(f"{ruta_guardar}{nombre_archivo}.csv", index=False)
        print(f"Se ha guardado el dataset en {ruta_guardar}{nombre_archivo}.csv")

    def to_excel(self, ruta_guardar: str, nombre_archivo: str, sheet_name: str = "Hoja1", sobreescribir: bool = False):
        ''' Guarda el dataset en formato excel.
        Args:
            ruta_guardar (str): Ruta donde se guardará el dataset.
            nombre_archivo (str): Nombre del archivo que se guardará.
            sheet_name (str): Nombre de la hoja del excel.
        '''
        ruta_total = f"{ruta_guardar}{nombre_archivo}.xlsx"
        if sobreescribir == False:
            # Comprobamos si el archivo ya existe
            if os.path.exists(ruta_total):
                print("El archivo ya existe.")
                seguir = input("¿Quieres sobreescribirlo? (y/n)")
                if seguir == "n":
                    return
        dataset = self.show()
        # guardamos toda la información
        for columna in QADataset.columnas:
            if columna is None:
                continue
            else:
                dataset[columna] = getattr(self, columna)
        dataset.to_excel(f"{ruta_guardar}{nombre_archivo}.xlsx", sheet_name=sheet_name, index=False)
        print(f"Se ha guardado el dataset en {ruta_guardar}{nombre_archivo}.xlsx")

    def crear_dataset_desde_excel(self, sheet_name) -> pd.DataFrame:
        ''' A través del excel, se crea un dataframe con las preguntas, respuestas, respuestas generadas y contexto para evaluarlo.
        Args:
            ruta_excel (str): Ruta del excel con las preguntas y respuestas.
                - Formato qa_excel: PROVEEDOR | PAG | EQUIPO | QUERY | QUERY_FILTRADA | ESPECIFICACIONES REALES	| PROVEEDOR | EQUIPO ELEGIDO | PDF | nº CARACTERISTICAS | nº ENCONTRADAS EN DATASHEET |	Nº REQ NO CUMPLIDOS
        '''       
        qa = pd.read_excel(self.ruta, sheet_name=sheet_name, dtype=str, header=0)
        qa_filtered = qa.dropna()
        dataset = pd.DataFrame()
        # Añadimos el id
        dataset['id'] = (qa_filtered['PROVEEDOR'] +"-"+ qa_filtered['PAG'] +"-"+ qa_filtered['EQUIPO'])
        dataset['id'] = dataset['id'].str.replace(" ", "_")
        dataset['question'] = [FormatoDato.limpiar_texto(query) for query in qa_filtered['QUERY FILTRADA']]
        dataset['ground_truth'] = qa_filtered['ESPECIFICACIONES REALES']
        dataset['equipo'] = qa_filtered['EQUIPO']
        dataset.reset_index(drop=True, inplace=True)
        return dataset

    def crear_dataset_desde_csv(self):
        ''' Carga el dataset desde un archivo csv.
        '''
        dataset = pd.read_csv(self.ruta)
        # Si el contexto está en formato string, lo convertimos a lista
        if ("contexts" in dataset.columns) and (type(dataset["contexts"][0]) == str):
            dataset["contexts"] = [literal_eval(contexto) for contexto in dataset["contexts"]]
        return dataset
    
    ### COMBINAR DATASETS

    def combinar(self, qa2):
        ''' Combina dos datasets en uno solo (inplace=True).
        Args:
            qa2 (QADataset): Segundo dataset a combinar.
        '''
        for columna in QADataset.columnas:
            col1 = getattr(self, columna)
            col2 = getattr(qa2, columna)
            setattr(self, columna, col1 + col2)
        print(f"Se han combinado los datasets.")
        return self

    ### AÑADIR METRICAS PARA EL DATASET
    def ragas(self, metricas: list):
        ''' Llamada a una lista de métricas de RAGAS.
        Args:
            - metrics (list): Lista de métricas de RAGAS.
        '''
        nombre_metricas = [metrica.name for metrica in metricas]
        dataset = self.to_dataset()
        resultado_raw = evaluate(dataset=dataset, metrics=metricas)
        resultado_pandas = resultado_raw.to_pandas()
        resultado = resultado_pandas.iloc[:, 4:]
        for columna in nombre_metricas:
            setattr(self, columna, resultado[columna].tolist())
        return resultado_raw
    
    def bertscore(self):
        ''' Usa BERTScore para evaluar las respuestas generadas. Se basa en la implementación de HuggingFace.'''
        answers = self.answer
        ground_truths = self.ground_truth
        scorer = BERTScorer(model_type='bert-base-uncased')
        precision_bertscore, recall_bertscore, f1_bertscore = scorer.score(answers, ground_truths)
        self.precision_bertscore = [float(p) for p in precision_bertscore]
        self.recall_bertscore = [float(r) for r in recall_bertscore]
        self.f1_bertscore = [float(f1) for f1 in f1_bertscore]
        print(f"BERTScore Precision: {precision_bertscore.mean():.4f}, Recall: {recall_bertscore.mean():.4f}, F1: {f1_bertscore.mean():.4f}")

    ### METRICAS PERSONALIZADAS
    @staticmethod
    def mismo_producto_individual(answer: str, ground_truth: str, modelo_llm: str="gpt-4o") -> bool:
        ### es el producto el mismo que se escogió? PODEMOS PASARLE EL MODELO COMO ATRIBUTO
        prompt_producto = '''Eres un experto en analizar textos y fichas sobre productos. Vas a comparar dos textos que describen productos y sus especificaciones. Sigue las siguientes instrucciones detalladamente para determinar si ambos textos se refieren al mismo producto:
        1. Compara el nombre del modelo del producto.
        2. Basado en esta comparación responde únicamente con un 1 si el modelo de ambos textos es el mismo o con un 0 si no se refieren al mismo producto o no lo sabes. Responde únicamente con un 1 o un 0. No añadas información sobre el answer o el ground_truth.

        Ejemplos: 
        answer: Nombre del equipo y proveedor: Neural Labs NL Ghost AI\n...
        ground_truth: NL GHOST AI procesa analítica...
        output: 1

        answer: Nombre del equipo y proveedor: Omniswitch 6800 de Alcatel Lucent\n...
        ground_truth: El switch elegido es el Industrial Ethernet serie 1000 de CISCO...
        output: 0

        answer: {answer}
        ground_truth: {ground_truth}
        output:
        '''

        llm = ChatOpenAI(temperature=0.7, model=modelo_llm)
        qa_prompt = ChatPromptTemplate.from_template(prompt_producto)
        llm_chain = qa_prompt | llm | StrOutputParser()

        output = llm_chain.invoke({"answer": answer, "ground_truth": ground_truth})
        if output != "1" and output != "0":
            print(output)
        if output == "No lo se": output = "0"
        return int(output)
    
    def es_mismo_producto(self, modelo_llm: str="gpt-3.5-turbo"):
        ''' Compara si el producto es el mismo que se escogió.
        '''
        mismo_producto = [QADataset.mismo_producto_individual(answer, ground_truth, modelo_llm) for answer, ground_truth in zip(self.answer, self.ground_truth)]
        self.mismo_producto = mismo_producto
        return mismo_producto
    
    @staticmethod
    def especificacion_es_correcta(sentencia: str, respuesta_verdadera: str, modelo_llm: str="gpt-3.5-turbo") -> float:
        ''' Compara si la sentencia se encuentra en el respuesta_verdadera.'''
        prompt_escorrecta = '''
        Eres un experto en analizar textos y fichas sobre productos. Tu tarea es comparar dos textos que describen productos y sus especificaciones para determinar si ambos se refieren al mismo producto.

        Instrucciones detalladas:

        1. Lee cuidadosamente la "Sentencia a buscar".
        2. Busca la "Sentencia a buscar" en el "Texto del producto".
        3. Responde únicamente con un "1" si la "Sentencia a buscar" se encuentra en el "Texto del producto", o con un "0" si no se encuentra.

        Sentencia a buscar:
        {sentencia}

        Texto del producto:
        {respuesta_verdadera}
        '''
        llm = ChatOpenAI(temperature=0, model=modelo_llm)
        qa_prompt = ChatPromptTemplate.from_template(prompt_escorrecta)
        llm_chain = qa_prompt | llm | StrOutputParser()

        output = llm_chain.invoke({"sentencia": sentencia, "respuesta_verdadera": respuesta_verdadera})
        if (output != "1") and (output != "0"):
            print("Que cosa mas rara")
            print(output)
            output = input("Introduce verdadero (1) o falso (0):")
        return int(output)

    # si le añadimos el metadata del num de especificaciones totales penalizamos respuestas incompletas
    @staticmethod
    def especificaciones_correctas_individual(answer: str, ground_truth: str, modelo_llm:str="gpt-3.5-turbo") -> float:
        ''' Te devuelve el porcentaje de especificaciones correctas comparando una respuesta
        con las especificaciones reales.'''
        sentencias = answer.split("\n")
        if len(sentencias) < 2:
            return 0
        else:
            sentencias = sentencias[1:]
        especificaciones_correctas = []
        for sen in sentencias:
            especificacion_correcta = QADataset.especificacion_es_correcta(sen, ground_truth, modelo_llm)
            especificaciones_correctas.append(especificacion_correcta)
        return sum(especificaciones_correctas) # nos devuelve el número total de espec. correctas, luego se puede dividir entre el num total del excel

    def son_especificaciones_correctas(self, modelo_llm: str="gpt-3.5-turbo"):
        ''' Compara un texto con otro para determinar el número de especificaciones que son correctas.
        '''
        especificaciones = [QADataset.especificaciones_correctas_individual(answer, ground_truth, modelo_llm) for answer, ground_truth in zip(self.answer, self.ground_truth)]
        self.especificaciones_correctas = especificaciones
        return especificaciones
    
    def son_especificaciones_correctas_norm(self, num_especificaciones):
        ''' Normaliza el número de especificaciones correctas en función del número total de especificaciones.
        '''
        especificaciones_correctas_norm  = self.especificaciones_correctas / num_especificaciones
        result = [1 if x > 1 else x for x in especificaciones_correctas_norm]
        self.especificaciones_correctas_norm = result
        return self.especificaciones_correctas_norm

    # pricing
    @staticmethod
    def num_tokens_from_messages(messages: dict, model: str="gpt-4o"):
        """Return the number of tokens used by a list of mensajes."""
        if model == "gpt-4o":
            encoding = tiktoken.get_encoding('o200k_base')
        else:
            encoding = tiktoken.get_encoding('cl100k_base')
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    @staticmethod
    def formato_contar_tokens(qa, row, mensaje_sistema):
        """ Devuelve el formato para que funcione la función num_tokens_from_messages
        Args:
            - qa: clase QADataset
            - row: int, fila del dataset
            - mensaje_sistema: str, mensaje de sistema utilizado
        Returns:
            - input: list, lista con el formato necesario para num_tokens_from_messages
            - output: list, lista con el formato necesario para num_tokens_from_messages
        """

        question = qa.question[row]
        answer = qa.answer[row]
        contexts = " ".join(qa.contexts[row])

        input = [{"mensaje_sistema": mensaje_sistema, "question": question, "contexts": contexts}]
        output = [{"answer": answer}]
        return input, output

    @staticmethod
    def precio_mensaje(input, output, model="gpt-4o"):
        """ Devuelve el precio de un mensaje
        Args:
            - input: list, lista con el formato necesario para num_tokens_from_messages
            - output: list, lista con el formato necesario para num_tokens_from_messages
        Returns:
            - precio: float, precio del mensaje en euros
        """
        num_tokens_input = QADataset.num_tokens_from_messages(input, model=model)
        num_tokens_output = QADataset.num_tokens_from_messages(output, model=model)

        price_input = 5 * num_tokens_input / 1000000
        price_output = 15 * num_tokens_output / 1000000

        price_total = price_input + price_output
        conversion_moneda = 1.07 # 1.07 USD = 1 EUR
        price_total_eur = price_total / conversion_moneda
        return price_total_eur

    def precio_mensajes(self, mensaje_sistema: str, model: str="gpt-4o"):
        """ Calcula el precio en euros de los mensajes del dataset en función del mensaje de sistema y el modelo utilizado"""
        qa = self.show()
        precio = 0
        for row in tqdm(range(len(self.question))):
            input, output = self.formato_contar_tokens(qa=qa, row=row, mensaje_sistema=mensaje_sistema)
            precio += self.precio_mensaje(input, output, model=model)
        return precio

    def show_metrics(self):
        ''' Muestra las métricas del dataset.
        '''
        dict_metricas = {columna: getattr(self, columna) for columna in QADataset.columnas_metricas if getattr(self, columna) is not None}
        metricas = pd.DataFrame(dict_metricas)
        return metricas
    
    def grafico(self, nombre: str, ruta_guardar: str, metricas: list = columnas_metricas):
        ''' Grafica las métricas del dataset.
        '''
        result = self.show_metrics()
        # ploteamos las cuatro ultimas columnas de resultados
        plt.figure(figsize=(10, 6))
        #### HACER QUE EMPIECE EN UNO EL INDICE #####
        plt.plot(range(1,1+len(result)), result[metricas], ".:", markersize=25)
        plt.xlabel('Número de pregunta')
        plt.ylabel('Valor')
        plt.title(f'Evaluación de {nombre}')
        plt.legend(metricas, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"{ruta_guardar}{nombre}.png", bbox_inches='tight')
        textstr = "**Valor medio de las métricas**\n"
        textstr += "\n".join([f"{key}: {value:.4f}" for key, value in zip(metricas, result[metricas].mean())])
        plt.gcf().text(-0.2, 0.5, textstr)
        plt.savefig(f"{ruta_guardar}Agregados_de_{nombre}.png", bbox_inches='tight')

    def __len__(self):
        ''' Número de preguntas (filas) en el dataset.'''
        return len(self.question)
    
