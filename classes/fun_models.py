path_root = r"C:/Users/jgpg000.edu/Desktop/LIBERAR TFM/"
# ruta clases personalizadas
import sys
path_classes = f"{path_root}classes"
sys.path.append(path_classes)

## LIBRERIAS
from pprint import pprint
import pandas as pd
import fitz
from class_data import *
from class_dataset import *
from tqdm import tqdm
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def error_valores_nulos():
    print("¡Revisa la lista de materiales!")
    raise Exception("Hay valores nulos en la lista de materiales")

def flatten(lista: list) -> list:
    ''' Aplana una lista de listas.
    Parámetros:
    - lista: lista de listas.
    Retorna:
    - lista_aplanada: lista aplanada.'''
    lista_aplanada = [item for sublist in lista for item in sublist]
    return lista_aplanada

def load_documents(path_excel: str, path_saved_json: str) -> list:
    ''' Carga la base de datos de documentos como objetos de la clase Documento.
    Parámetros:
    - path_excel: ruta del archivo Excel con la lista de materiales.
    - path_saved_json: ruta donde se encuentran los documentos en formato JSON.
    Retorna:
    - documentos: lista de objetos de la clase Documento.'''
    lista_material = pd.read_excel(path_excel, dtype=str, header=0).to_numpy()
    if len(lista_material[lista_material == "nan"]) > 0:
        error_valores_nulos()

    # Cargamos documentos
    documentos = []
    for equipo, proveedor, modelo, pdfs, proyecto in tqdm(lista_material):
        ruta_cargar = f"C:/Users/jgpg000.edu/Desktop/Dataset/proveedores/{proyecto}/"
        if (pdfs == None) or (type(pdfs) is not str): continue
        documento = Documento.cargar(path_saved_json, ruta_cargar, pdfs)
        documentos.append(documento)
    print(f"Se han cargado {len(documentos)} documentos.")
    return documentos

def create_documents(path_excel: str, path_saved_json: str, items_folder: str) -> list:
    ''' Crea la base de datos de documentos como objetos de la clase Document 
    y los guarda en formato JSON.
    Parámetros:
    - path_excel: ruta del archivo Excel con la lista de materiales.
    - path_saved_json: ruta donde se guardarán los documentos en formato JSON.
    - items_folder: carpeta donde se encuentran los documentos PDF.
    Retorna:
    - documentos: lista de objetos de la clase Documento.
    '''
    lista_material = pd.read_excel(path_excel, dtype=str, header=0).to_numpy()
    if len(lista_material[lista_material == "nan"]) > 0:
        error_valores_nulos()

    # Cargamos documentos
    documentos = []
    prompt_sistema = Tabla.prompt_sistema
    for equipo, proveedor, modelo, pdfs, proyecto in tqdm(lista_material):
        ruta_producto = f"{items_folder}/{proyecto}/"
        documento = Documento(ruta_producto, pdfs, modelo, proveedor, equipo)
        if (pdfs == None) or (type(pdfs) is not str): continue
        documento.resumir_tablas(n=1, model= "gpt-4o", prompt_sistema=prompt_sistema)
        documento.guardar(path_saved_json)
        documentos.append(documento)
    print(f"Se han creado {len(documentos)} documentos.")
    return documentos

## Creacion documentos de Langchain
from langchain_core.documents import Document

def load_documents_langchain(documentos: list, anadir_tablas: bool= True, verbose: bool=True) -> list:
    ''' A partir de una lista de documentos de la clase Documento, 
    crea una lista de documentos de la clase Document de Langchain.
    
    Parámetros:
    - documentos: lista de objetos de la clase Documento.
    - anadir_tablas: si es True, se añaden los resúmenes de tablas.
    - verbose: si es True, imprime un mensaje de confirmación.

    Retorna:
    - documentos_langchain: lista de objetos de la clase Document de Langchain.
    '''
    # Convertimos en Langchain
    documentos_langchain = []
    # Create a document from rows in archivo_textos
    for documento in tqdm(documentos):
        metadata = documento.show()
        documento_langchain = Document(documento.texto_traducido,
                        metadata={
                            key: value for key, value in metadata.items() if key in ['nombre_pdf', 'modelo', 'proveedor', 'equipo']
                            }
                        )
        documentos_langchain.append(documento_langchain)
        if documento.resumenes is None:
            continue
        if anadir_tablas:
            for resumen_tabla in documento.resumenes:
                resumen_tabla_langchain = Document(resumen_tabla,
                            metadata={
                                key: value for key, value in metadata.items() if key in ['nombre_pdf', 'modelo', 'proveedor', 'equipo']
                                }
                            )
                resumen_tabla_langchain.metadata["tabla"] = True
                documentos_langchain.append(resumen_tabla_langchain)
    if verbose:
        print(f'''Se han generado {len(documentos_langchain)} documentos de Langchain.
    De ellos, {len(documentos)} son el texto y {len(documentos_langchain) - len(documentos)} son resúmenes de tablas.''')
    return documentos_langchain


from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

### FUNCION DE FILTRO ###
def rag_filter(vectorstore, equipo: str, template_system: str, model_llm: str = "gpt_4o", search_type: str = "mmr", search_kwargs: dict = {"k": 8, "fetch_k": 40}):
    ''' Añade el filtro de equipo al RAG
    Parámetros:
    - vectorstore: Objeto de la clase FAISS.
    - equipo (str): Nombre del equipo.
    - template_system (str): Plantilla del sistema.
    - model_llm (str): Modelo de lenguaje de OpenAI.
    - search_type (str): Tipo de búsqueda.
    - search_kwargs (dict): Argumentos de búsqueda.
    Devuelve:
    - rag_filtered: Modelo con el filtro de equipo incluido.
    '''
    search_kwargs = search_kwargs.copy()
    search_kwargs["filter"] = {"equipo": equipo}
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    rag_filtered = create_RAG(retriever=retriever,
                            plantilla_RAG=template_system,
                            model=model_llm)
    return rag_filtered

### FUNCIONES PARA EL RAG ###

def format_docs(docs):
    formateado = ""
    for i, doc in enumerate(docs):
        formateado += f"[Chunk {i}]"
        formateado += f"{doc.page_content} "
    return formateado

def retrieve_answer(output):
    return output.content

def metadata_modelos(docs: list) -> str:
    datos_extra = ""
    for i, doc in enumerate(docs):
        datos_extra += f"[Datos Chunk {i}]"
        datos_extra += f"{doc.metadata['proveedor']} "
        datos_extra += f"{doc.metadata['modelo']} "
    return datos_extra

def create_database(ruta_guardar_vectores: str, fragmentos_texto: list, nombre_embedding: str="text-embedding-3-small", verbose: bool=False):
    embedding_model = OpenAIEmbeddings(model=nombre_embedding)
    store = LocalFileStore(ruta_guardar_vectores)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding_model, store, namespace=f"openai-{nombre_embedding}")
    vectorstore = FAISS.from_documents(fragmentos_texto, cached_embedder)
    if verbose: print(f"Espacio creado correctamente. Contiene {vectorstore.index.ntotal} vectores.")
    return vectorstore

def create_retriever(vectorstore, search_type: str="mmr", search_kwargs: dict={"k": 8, "fetch_k": 40}, verbose: bool=False):
    if verbose: print("Retriever creado correctamente.")
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

def create_RAG(retriever, model, plantilla_RAG, verbose=False):
    '''
    Crea un RAG a partir de un retriever y un modelo de lenguaje.
    Parámetros:
    - retriever: objeto de la clase Retriever.
    - model: modelo de lenguaje de OpenAI.
    - plantilla_RAG: plantilla de ChatPromptTemplate.
    - verbose: si es True, imprime un mensaje de confirmación.
    '''
    qa_prompt = ChatPromptTemplate.from_template(plantilla_RAG)
    llm = ChatOpenAI(temperature=0, model=model)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["contexts"])))
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"contexts": retriever , "question": FormatoDato.limpiar_texto, "metadata": retriever | metadata_modelos},
    ).assign(answer=rag_chain_from_docs)
    if verbose: 
        print("""RAG creado correctamente.
        Llámalo con <rag_name>.invoke(<query>)""")
    return rag_chain_with_source