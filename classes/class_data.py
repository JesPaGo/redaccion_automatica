import pandas as pd
import numpy as np
import re
import fitz
import json
from openai import OpenAI
from os import path
import random
from langdetect import detect


# TO DO LIST (mirar donde con los comentarios):
# - añadir precio como variable obligatoria: generamos los PRECIOS con una variable aleatoria gaussiana: N(1000, 300)

######################################################################################

class FormatoDato:
    # Guarda características generales de una tabla o texto
    def __init__(self, dato: "pd.DataFrame", nombre_documento: str, equipo: str, modelo: str, proveedor: str, precio: float=None):
        self.dato = dato
        self.nombre_documento = nombre_documento
        self.equipo = equipo
        self.modelo = modelo
        self.proveedor = proveedor
        if precio == None: # precio como variable para jugar
            self.precio = precio
        else:
            self.precio = round(random.gauss(1000, 300),2)
    
    @staticmethod
    def limpiar_texto(texto: str) -> str:
        ''' Reemplaza los marcadores y limpia el texto de caracteres no deseados y saltos de línea.
        Parametros:
            texto (str): texto a limpiar.
        Devuelve:
            texto_limpio (str): texto limpio.
        '''
        sen = FormatoDato.reemplazar_marcadores(texto) # reemplazamos marcadores
        sen = sen.replace("-\n","")
        sen2 = sen.replace("\n"," ").replace("\t"," ").replace('\xad ','').replace('\uf0b2',' ').strip()
        sen3 = sen2.replace('✓','').replace('\xad','').replace("®"," ").replace('\x07', ' ').replace("\xa0","").strip() # quitamos saltos de linea
        texto_limpio = re.sub(r'\s{2,}', ' ', sen3) # quitamos espacios en blanco
        return texto_limpio
    
    @classmethod
    def reemplazar_marcadores(cls, texto: str) -> str:
        ''' Enumera una cadena de texto con marcadores de lista y sublista.
        Args:
            texto (str): texto a enumerar.
        '''
        marcadores = ["\uf0b7", ". o", "•", "", "·", "\u00B7", "- "] # voy a ir añadiendo todos los que me encuentres de las queries del dataset
        marcadores_nuevos = [" [LISTA] ", " [SUBLISTA] ", " [SUBSUBLISTA]"]
        texto = re.sub(r'\s{2,}', ' ', texto) # elimino espacios en blanco

        comienzo_marcadores = {marca: texto.find(marca) for marca in marcadores if texto.find(marca) != -1}
        marcadores_final = {}
        for marca_nueva in marcadores_nuevos:
            if comienzo_marcadores == {}: break # si no hay marcadores, salgo del bucle
            primera_marca = min(comienzo_marcadores, key=comienzo_marcadores.get)
            marcadores_final[primera_marca] = marca_nueva
            if primera_marca == ". o": # si es punto y seguido, añado el punto y seguido
                marcadores_final[primera_marca] = "." + marca_nueva
            del comienzo_marcadores[primera_marca]

        # reemplazo los marcadores
        for marca, nueva_marca in marcadores_final.items():
            texto = texto.replace(marca, nueva_marca)

        texto = re.sub(r'\s{2,}', ' ', texto)
        return texto

######################################################################################

class Tabla(FormatoDato):
    ''' Lista de métodos de la clase:
    - procesar: Procesa la tabla para convertirla en texto (str).
    - resumen_tabla (classmethod): Resumen de la tabla en texto (str).
    - resumen: Crea un objeto Texto usando el resumen de una tabla.
    - show: Muestra la salida en formato diccionario.
    '''
    prompt_sistema = "Voy a proporcionarte un cuadro de datos con información técnica sobre varios modelos. Necesito que resumas esta información en español, manteniendo los términos técnicos en inglés. Cada modelo debe ser descrito por separado. Asegúrate de incluir todas las especificaciones técnicas importantes. No utilices formatos de texto como negritas, cursivas, listas, o tablas. El resumen debe ser conciso y preciso, cubriendo todos los detalles técnicos esenciales sin omitir ninguno. No agregues información adicional ni comentarios personales. Solo incluye el resumen solicitado."
    
    def __init__(self, dato: "pd.DataFrame", pagina: int, numero_tabla: int, nombre_documento: str, equipo: str, modelo: str, proveedor: str, precio: float=None):
        super().__init__(dato, nombre_documento, equipo, modelo, proveedor, precio)
        self.pagina = pagina
        self.numero_tabla = numero_tabla
        self.tabla_procesada = self.procesar()

    @staticmethod
    def limpiar_tabla(tabla: pd.DataFrame) -> pd.DataFrame:
        ''' Preprocesamiento que se le hace a las tablas en formato. Eliminamos los bullets y los saltos de linea.
        Se hace a nivel de DataFrame de Pandas'''
        tabla = tabla.replace({"\n": " ","•": "", "\t": " "}, regex=True)
        return tabla

    @staticmethod
    def rellenar_tabla_por_columnas(tabla: pd.DataFrame) -> pd.DataFrame:
        ''' Funciona a nivel de columnas. Rellena los nombres de las columnas sin título. Además de los "subnombres" de tablas que se
        han rellenado con None.
        FALLA SI LA CELDA DE ARRIBA ESTÁ VACÍA.
        '''
        for columns in tabla.columns:
            for i in range(1, len(tabla[columns])):
                valor_previo = str(tabla[columns][i-1])
                value = str(tabla[columns][i])
                if re.search(r"^Col\d", value):
                    tabla.loc[i, columns] = valor_previo
                if (value == "None") or (value == ""):
                    tabla.loc[i, columns] = valor_previo
        tabla = tabla.replace({"None": "", None: ""})
        return tabla

    @staticmethod
    def pandas_a_texto(tabla: pd.DataFrame, atributos: "pd.Series") -> list:
        ''' Convierte un DataFrame de Pandas a texto. Se usa para normalizar el texto de las tablas.
        Como entrada tiene el df procesado y los nombres de las columnas de la tabla.
        Añade como primera sentencia el nombre
        Lee las tablas por filas.'''
        tabla_texto = []
        for index, row in tabla.iterrows():
            raw_row_str = ""
            for n_col, col in enumerate(tabla.columns):
                raw_row_str += f"{atributos[n_col]}: {row[col]}. "
            raw_row_str = raw_row_str.replace("..",".").replace("\n", " ").replace("\t", " ")
            row_str = re.sub(r"\s{2,}", " ", raw_row_str).strip()
            tabla_texto.append(row_str)
        return tabla_texto

    def tabla_con_metadatos(self, texto_tabla: list) -> pd.DataFrame:
        '''Usando como elemento de ENTRADA el diccionario generado para cada tabla,
        construimos un string añadiendo los metadatos de la tabla al principio de ella.'''
        tabla_metadatos = ""
        tabla_metadatos += f"[MATERIAL] {self.equipo} "
        tabla_metadatos += f"[MODELO] {self.modelo} "
        tabla_metadatos += f"[PROVEEDOR] {self.proveedor} "
        tabla_metadatos += f"[TABLA] {texto_tabla}"
        return tabla_metadatos

    def procesar(self) -> str:
        tabla = self.dato
        # Construimos la tabla uniendo añadiendo las columnas como la primera fila
        df = Tabla.limpiar_tabla(tabla)
        atributos = pd.DataFrame([df.columns.tolist()], columns=df.columns)
        new_df = pd.concat([atributos, df], ignore_index=True)
        # Eliminamos el índice
        dfT = new_df.T.reset_index().drop("index", axis=1)
        # Completamos nombres de columnas y eliminamos None
        dfT = Tabla.rellenar_tabla_por_columnas(dfT)
        atributos = dfT[0] # nombres de las columnas
        # Volvemos a transponer la tabla para leer de fila en fila (df = df.T.T)
        df = dfT.T
        ### Rellenamos por filas para quitar huecos en blanco ###
        df = Tabla.rellenar_tabla_por_columnas(df)
        df = df.drop(0, axis=0)
        # Convertimos la tabla (pd.DataFrame) a una lista texto
        texto_tabla = Tabla.pandas_a_texto(df, atributos)
        if texto_tabla == []: 
            return None # No incluimos tablas vacías
        tabla_final = self.tabla_con_metadatos(texto_tabla)
        return tabla_final
    
    @classmethod
    def resumen_tabla(cls, tabla: str, n: int=1, model: str="gpt-4o", prompt_sistema: str=""):
        client = OpenAI()
        response = client.chat.completions.create(
          model=model,
          messages=[
            {
              "role": "system",
              "content": f"{prompt_sistema}"
              },
            {
              "role": "user",
              "content": f"{tabla}"
            }
          ],
          temperature=1,
          max_tokens=2056,
          top_p=0.8,
          frequency_penalty=0,
          presence_penalty=0,
          n=n # numero de respuestas
        )
        texto_resumen = {}
        for i in range(len(response.choices)):
            texto = response.choices[i].message.content
            texto_limpio = FormatoDato.limpiar_texto(texto)
            if texto_limpio == "": texto_limpio = "NULL"
            texto_resumen[i] = texto_limpio
        return texto_resumen

    @staticmethod
    def texto_plano_resumen_tablas(resumenes_tabla: dict) -> str:
        ''' A partir de una estructura de TABLA RESUMIDA, devuelve una
        string concatenando los resumenes de una tabla.'''
        texto_tablas = ""
        for texto in resumenes_tabla.values():
            texto_tablas += " [TABLA] "
            texto_tablas += texto
        return texto_tablas.strip()

    def resumen(self, n: int=1, model: str="gpt-4o", prompt_sistema: str=""):
        ''' Crea un objeto Texto a partir del resumen de una tabla'''
        tabla = self.dato
        texto_resumen_dict = Tabla.resumen_tabla(tabla, n, model, prompt_sistema)
        texto_resumen = Tabla.texto_plano_resumen_tablas(texto_resumen_dict)
        return Texto(self.dato, self.pagina, self.nombre_documento, self.equipo, self.modelo, self.proveedor, self.precio, texto_resumen, tabla=True)

    def show(self):
        '''Muestra la salida en formato diccionario.'''
        return {"dato_raw": self.dato,
            "nombre_documento": self.nombre_documento,
            "pagina": self.pagina,
            "numero_tabla": self.numero_tabla,
            "equipo": self.equipo, #mirar si añadir lo que es cada objeto
            "modelo": self.modelo,
            "proveedor": self.proveedor,
            "precio": self.precio,
            "tabla_procesada": self.tabla_procesada
            }

######################################################################################

class Texto(FormatoDato):
    ''' Lista de métodos de la clase:
    - texto: Devuelve el texto procesado.
    - show: Muestra la salida en formato diccionario.
    '''
    def __init__(self, dato, pagina: int, nombre_documento: str, equipo: str, modelo: str, proveedor: str, precio: float=None, texto_procesado: str=None, tabla: bool=False):
        super().__init__(dato, nombre_documento, equipo, modelo, proveedor, precio)
        self.pagina = pagina
        if type(self.dato) is not pd.DataFrame:
            self.texto_procesado = self.procesar()

    @property
    def texto(self):
        return self.texto_procesado

    def show(self):
        '''Muestra la salida en formato diccionario.'''
        return {"dato_raw": self.dato,
            "nombre_documento": self.nombre_documento,
            "pagina": self.pagina,
            "equipo": self.equipo, #mirar si añadir lo que es cada objeto
            "modelo": self.modelo,
            "proveedor": self.proveedor,
            "precio": self.precio,
            "texto_procesado": self.texto_procesado
            }
    
    bound_x_upper = 15
    bound_x_lower = 550
    bound_y_upper = 45
    bound_y_lower = 800
    @staticmethod
    def limit_z(z0: float, z1: float, z_upper: float, z_lower: float) -> bool:
        ''' Devuelve True si el chunk ha superado los límites del documento en un eje.'''
        return not((z_upper <= z0 <= z_lower) and (z_upper <= z1 <= z_lower))
    
    @staticmethod
    def eliminar_margenes(lista_bloques_texto: list, bound_x_upper=bound_x_upper, bound_x_lower=bound_x_lower,
                       bound_y_upper=bound_y_upper, bound_y_lower=bound_y_lower) -> list:
        ''' Elimina los chunks que se encuentran fuera de los márgenes.
        El formato devuelto por extraer_texto es:
            (x0, y0, x1, y1, Texto, nº bloque, ¿es_foto?)'''
        parrafos = []
        for parse in lista_bloques_texto:
            x0, y0, x1, y1 = parse[:4]
            if Texto.limit_z(x0, x1, bound_x_upper, bound_x_lower) or Texto.limit_z(y0, y1, bound_y_upper, bound_y_lower):
                continue
            parrafos.append(parse[4]) 
        return parrafos

    def procesar(self):
        texto_pagina = self.dato
        if type(texto_pagina) is pd.DataFrame: # No se ejecuta para las tablas
            return None
        texto_procesado = Texto.eliminar_margenes(texto_pagina)
        texto_procesado = " ".join(texto_procesado)
        texto_procesado = FormatoDato.limpiar_texto(texto_procesado)
        return texto_procesado
    
######################################################################################
from langdetect import detect
from deep_translator import GoogleTranslator

class Documento:
    ''' La clase Documento recoge las tablas y textos de un documento PDF. Incluye los métodos:
    - show: Muestra la salida en formato diccionario.
    - texto (getter): Devuelve el texto procesado.
    - tablas (getter): Devuelve una lista de tablas procesadas.
    - cargar: Carga un documento en formato json.
    - guardar: Guarda el documento en un archivo json.
    '''
    def __init__(self, ruta: str, nombre_pdf: str, modelo: str, proveedor: str, equipo: str, precio: float=None, texto: str=None, tablas: list=None, resumenes: list=None, texto_traducido: str= None):
        self.nombre_pdf = nombre_pdf
        self.modelo = modelo
        self.proveedor = proveedor
        self.equipo = equipo
        if precio is None:
            self.precio = round(random.gauss(1000, 300),2) # usamos el precio para jugar con las limitaciones del rag
        else:
            self.precio = precio
        self.documento = self.abrir_documento(f"{ruta}{nombre_pdf}.pdf")
        self.n_paginas = self.documento.page_count
        self.__resumenes = resumenes
        # Cargamos las tablas únicamente si se han pasado como argumento
        if tablas is None:
            self.__tablas = self.extraer_todo_tabla()
        else:
            self.__tablas = tablas
        # Cargamos el texto únicamente si se ha pasado como argumento
        if texto is None:
            self.__texto_paginas = self.extraer_todo_texto()
            self.__texto = self.extraer_texto()
        else:
            self.__texto_paginas = None
            self.__texto = texto
        if texto_traducido is None:
            self.__texto_traducido = self.traducir_texto()
        else:
            self.__texto_traducido = texto_traducido

    def abrir_documento(self, ruta: str) -> fitz.Document:
        ''' Abre un documento PDF y lo guarda como un objeto fitz.Document.'''
        documento = fitz.open(ruta)
        return documento
    
    def extraer_texto(self) -> str:
        return " ".join([texto.texto for texto in self.__texto_paginas])

    def extraer_todo_texto(self) -> list:
        texto_paginas = []
        for n in range(self.n_paginas):
            pagina = self.documento[n]
            texto_pagina = pagina.get_text("blocks")
            texto_paginas.append(Texto(texto_pagina, n, self.nombre_pdf, self.equipo, self.modelo, self.proveedor, self.precio))
        return texto_paginas

    def extraer_todo_tabla(self) -> list:
        tablas = []
        for n in range(self.n_paginas):
            pagina = self.documento[n]
            tablas_por_pagina = pagina.find_tables()
            for i, tabla in enumerate(tablas_por_pagina):
                tabla_pagina = tabla.to_pandas()
                tabla = Tabla(tabla_pagina, n, i, self.nombre_pdf, self.equipo, self.modelo, self.proveedor, self.precio)
                if tabla.tabla_procesada is not None:
                    tablas.append(tabla.tabla_procesada)
        return tablas

    def esta_en_espanol(self) -> bool:
        ''' Comprueba si el texto está en español.'''
        languaje = detect(self.__texto)
        return languaje == "es"
    
    @staticmethod
    def dividir_texto(texto, n=2500) -> list:
        ''' Divide el texto en trozos de n caracteres.
        El máximo de caracteres permitidos por GoogleTranslator es 5000.
        '''
        return [texto[i:i + n] for i in range(0, len(texto), n)]

    def traducir_texto(self) -> str:
        ''' Traduce el texto a español si está en inglés.'''
        lang_esp = self.esta_en_espanol()
        if lang_esp == True:
            return self.__texto
        else:
            texto_dividido = Documento.dividir_texto(self.__texto)
            texto_traducido = [GoogleTranslator(source='en', target='es').translate(string) for string in texto_dividido]
            return " ".join(texto_traducido)

    def show(self) -> dict:
        '''Muestra la salida en formato diccionario.'''
        return {"nombre_pdf": self.nombre_pdf,
            "modelo": self.modelo,
            "proveedor": self.proveedor,
            "equipo": self.equipo,
            "precio": self.precio,
            "n_paginas": self.n_paginas,
            "tablas": self.__tablas,
            "resumenes": self.__resumenes,
            "texto": self.__texto,
            "texto_traducido": self.__texto_traducido
            }

    @staticmethod
    def cargar(ruta_guardado: str, ruta_documento: str, nombre_pdf: str):
        ''' Carga un documento en formato json.'''
        # comprobamos si existe el archivo en la ruta
        ruta_completa = f"{ruta_guardado}{nombre_pdf}.json"
        if path.exists(ruta_completa) == False: # si no existe avisamos
            print(f"[WARNING] No existe el archivo {nombre_pdf}.")
        else:
            with open(ruta_completa, "r") as f:
                datos_documento = json.load(f)
            f.close()
            return Documento(ruta_documento, nombre_pdf, datos_documento['modelo'], datos_documento['proveedor'], datos_documento['equipo'], datos_documento['precio'], datos_documento['texto'], datos_documento['tablas'], datos_documento['resumenes'], datos_documento['texto_traducido'])

    def guardar(self, ruta: str, nombre: str = None, sobreescribir: bool = False):
        ''' Guarda el documento en un archivo json.'''
        # si el nombre no se pasa como argumento, se guarda con el nombre original
        if nombre is None:
            nombre = self.nombre_pdf
        # comprobamos si ya existe el archivo
        ruta_completa = f"{ruta}{nombre}.json"
        if sobreescribir == False and path.exists(ruta_completa): # si no se quiere sobreescribir y el archivo ya existe avisamos
            print(f"[WARNING] El archivo {nombre} ya está guardado en la ruta.") 
        else: # si no existe o se quiere sobreescribir, guardamos el archivo
            with open(f"{ruta}{nombre}.json", "w") as f:
                json.dump(self.show(), f)
            f.close()
            print(f"Documento {self.nombre_pdf} guardado correctamente.")

    def __str__(self):
        return f"El documento {self.nombre_pdf}, con {self.n_paginas} páginas, contiene {len(self.tablas)} tablas y {len(self.texto.split())} palabras."
    
    def resumir_tablas(self, n: int=1, model: str="gpt-4o", prompt_sistema: str="", sobreescribir: bool=False):
        ''' Resumen de todas las tablas del documento.'''
        if self.__tablas is None: 
            return print("No hay tablas en el documento.")
        if sobreescribir:
            resumenes_tablas = []
            for tabla in self.tablas:
                resumen_tabla = Tabla.resumen_tabla(tabla, n, model, prompt_sistema)
                texto_resumen_tabla = Tabla.texto_plano_resumen_tablas(resumen_tabla)
                resumenes_tablas.append(texto_resumen_tabla)
            print("Resumenes reescritos.")
            self.__resumenes = resumenes_tablas
        else:    
            if self.__resumenes is not None: 
                print("Ya se han resumido las tablas.")
            else:
                resumenes_tablas = []
                for tabla in self.tablas:
                    resumen_tabla = Tabla.resumen_tabla(tabla, n, model, prompt_sistema)
                    texto_resumen_tabla = Tabla.texto_plano_resumen_tablas(resumen_tabla)
                    resumenes_tablas.append(texto_resumen_tabla)
                self.__resumenes = resumenes_tablas

    @property
    def resumenes(self):
        return self.__resumenes

    @property
    def texto(self):
        return self.__texto
    
    @property
    def texto_traducido(self):
        return self.__texto_traducido
    
    @property
    def tablas(self):
        return self.__tablas

######################################################################################

def reemplazar_marcadores(query: str) -> str:
    ''' Enumera una query con marcadores de lista y sublista.
    Args:
        query (str): texto a enumerar.
    '''
    marcadores = ["\uf0b7", ". o", "•", ""] # voy a ir añadiendo todos los que me encuentres de las queries del dataset
    marcadores_nuevos = [" [LISTA] ", " [SUBLISTA] "]
    query = re.sub(r'\s{2,}', ' ', query) # elimino espacios en blanco

    comienzo_marcadores = {marca: query.find(marca) for marca in marcadores if query.find(marca) != -1}
    marcadores_final = {}
    for marca_nueva in marcadores_nuevos:
        if comienzo_marcadores == {}: break # si no hay marcadores, salgo del bucle
        primera_marca = min(comienzo_marcadores, key=comienzo_marcadores.get)
        marcadores_final[primera_marca] = marca_nueva
        if primera_marca == ". o": # si es punto y seguido, añado el punto y seguido
            marcadores_final[primera_marca] = "." + marca_nueva
        del comienzo_marcadores[primera_marca]

    # reemplazo los marcadores
    for marca, nueva_marca in marcadores_final.items():
        query = query.replace(marca, nueva_marca)

    query = re.sub(r'\s{2,}', ' ', query)
    return query

######################################################################################