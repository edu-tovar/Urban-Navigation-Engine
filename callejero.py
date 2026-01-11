"""
callejero.py

Matemática Discreta - IMAT
ICAI, Universidad Pontificia Comillas

Grupo: GP08B
Integrantes:
    - Alejandro Royo-Villanova Seguí
    - Eduardo Tovar Ruíz

Descripción:
Librería con herramientas y clases auxiliares necesarias para la representación de un callejero en un grafo.

Complétese esta descripción según las funcionalidades agregadas por el grupo.
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from difflib import get_close_matches
import os
from typing import Tuple
import matplotlib.pyplot as plt


STREET_FILE_NAME="direcciones.csv"

PLACE_NAME = "Madrid, Spain"
MAP_FILE_NAME="madrid.graphml"

MAX_SPEEDS={'living_street': '20',
 'residential': '30',
 'primary_link': '40',
 'unclassified': '40',
 'secondary_link': '40',
 'trunk_link': '40',
 'secondary': '50',
 'tertiary': '50',
 'primary': '50',
 'trunk': '50',
 'tertiary_link':'50',
 'busway': '50',
 'motorway_link': '70',
 'motorway': '100'}


class ServiceNotAvailableError(Exception):
    "Excepción que indica que la navegación no está disponible en este momento"
    pass


class AdressNotFoundError(Exception):
    "Excepción que indica que una dirección buscada no existe en la base de datos"
    pass


############## Parte 2 ##############


def coord_to_decimal(cadena:str)-> float:
    """Función que convierte una coordenada en formato grados, minutos y segundos a grados decimales

    Args:
        cadena (str): Coordenada en formato DMS (grados, minutos, segundos) seguida de la orientación (N, S, E o W)

    Returns:
        float: Valor de la coordenada expresada en grados decimales
    """
    coord = cadena.replace("°"," ").replace("'", " ").replace("''", "").strip()
    grados, minutos, segundos, orientacion = coord.split()
    num = int(grados) + int(minutos)/60 + float(segundos)/3600
    if orientacion in ("S","W"):
        num = -num
    return num


def construir_direccion(fila: pd.Series) -> str:
    """Función que construye una dirección a partir de los campos del callejero

    Args:
        fila (pd.Series): Fila del DataFrame del callejero que contiene los campos "VIA_CLASE", "VIA_PAR", "VIA_NOMBRE" y "NUMERO"

    Returns:
        str: Dirección completa construida a partir de los datos de la fila
    """
    # Cojo cada parte del nombre de la calle
    via_clase = str(fila["VIA_CLASE"]).title()
    via_par = fila["VIA_PAR"]
    via_nombre = str(fila["VIA_NOMBRE"]).title()
    numero = str(fila["NUMERO"]).strip()
    # Vamos añadiendo las distintas palabras que forman el nombre de la calle
    nombre_final = [via_clase]
    # VIA_PAR puede ser NaN
    if isinstance(via_par, str):
        nombre_final.append(via_par.strip().lower())
    nombre_final.append(via_nombre)
    # Juntamos todas las partes con espacios
    nombre_via = " ".join(nombre_final)
    return f"{nombre_via}, {numero}"


def carga_callejero() -> pd.DataFrame:
    """ Función que carga el callejero de Madrid, lo procesa y devuelve
    un DataFrame con los datos procesados
    
    Args: None
    Returns:
        DataFrame: dataframe con los datos del callejero procesados.
    Raises:
        FileNotFoundError si el fichero csv con las direcciones no existe
    """
    # Columnas que pide el enunciado
    COLUMNAS = ["VIA_CLASE", "VIA_PAR", "VIA_NOMBRE", "NUMERO", "LATITUD", "LONGITUD"]
    try:
        direcciones_df = pd.read_csv("direcciones.csv", sep=";", encoding="latin-1", usecols=COLUMNAS)
    except FileNotFoundError:
        raise FileNotFoundError("El fichero csv con las direcciones no existe")
    # Convertimos de coordenadas a grados decimales
    direcciones_df["LATITUD"] = direcciones_df["LATITUD"].map(coord_to_decimal)
    direcciones_df["LONGITUD"] = direcciones_df["LONGITUD"].map(coord_to_decimal)
    direcciones_df["VIA_CLASE"] = direcciones_df["VIA_CLASE"].str.upper()
    direcciones_df["VIA_PAR"]   = direcciones_df["VIA_PAR"].str.upper()
    direcciones_df["VIA_NOMBRE"] = direcciones_df["VIA_NOMBRE"].str.upper()
    direcciones_df["DIRECCION_TEXTO"] = direcciones_df.apply(construir_direccion,axis=1)
    return direcciones_df


def busca_direccion(direccion:str, callejero:pd.DataFrame) -> Tuple[float,float]:
    """ Función que busca una dirección, dada en el formato
        calle, numero
    en el DataFrame callejero de Madrid y devuelve el par (latitud, longitud) en grados de la
    hubicación geográfica de dicha dirección
    
    Args:
        direccion (str): Nombre completo de la calle con número, en formato "Calle, num"
        callejero (DataFrame): DataFrame con la información de las calles
    Returns:
        Tuple[float,float]: Par de float (latitud,longitud) de la dirección buscada, expresados en grados
    Raises:
        AdressNotFoundError: Si la dirección no existe en la base de datos
    Example:
        busca_direccion("Calle de Alberto Aguilera, 23", data)=(40.42998055555555,-3.7112583333333333)
        busca_direccion("Calle de Alberto Aguilera, 25", data)=(40.43013055555555,-3.7126916666666667)
    """
    direccion_busqueda = direccion.strip()
    # Buscamos coincidencia en la columna DIRECCION_TEXTO
    coincidencia = callejero["DIRECCION_TEXTO"] == direccion_busqueda
    # Si no hay lanzamos error
    if not coincidencia.any():
        raise AdressNotFoundError(f"La dirección {direccion} no existe en el callejero")
    # Si hay varias filas que coinciden, elegimos la primera
    fila = callejero[coincidencia].iloc[0]
    latitud = float(fila["LATITUD"])
    longitud = float(fila["LONGITUD"])
    return latitud, longitud


def busca_direccion_fuzzy(direccion:str, callejero:pd.DataFrame, umbral_similitud:float=0.8) -> Tuple[float,float]:
    """ Función que busca la dirección más parecida a la dada, lo cual lo hace con comparación aproximada de cadenas (fuzzy search),
    dada en el formato calle, numero
    en el DataFrame callejero de Madrid y devuelve el par (latitud, longitud) en grados de la
    hubicación geográfica de dicha dirección
    
    Args:
        direccion (str): Nombre completo de la calle con número, en formato "Calle, num"
        callejero (DataFrame): DataFrame con la información de las calles
        umbral_similitud (float): Valor entre 0 y 1. Solo se aceptan coincidencias cuya similitud sea al menos este valor.
    Returns:
        Tuple[float,float]: Par de float (latitud,longitud) de la dirección buscada, expresados en grados
    Raises:
        AdressNotFoundError: Si no se encuentra ninguna dirección lo bastante parecida en la base de datos
    Example:
        busca_direccion_fuzzy("Cll de Alberto Aguiler, 23", data)=(40.42998055555555,-3.7112583333333333)
        busca_direccion_fuzzy("Calle de Alberto Aguilera, 23", data)=(40.42998055555555,-3.7112583333333333)
    """
    direccion_busqueda = direccion.strip()
    # Lista de direcciones del dataframe
    direcciones_lista = callejero["DIRECCION_TEXTO"].tolist()
    # Buscamos la dirección más parecida (n=1)
    coincidencias = get_close_matches(direccion_busqueda, direcciones_lista, n=1, cutoff=umbral_similitud)
    if not coincidencias:
        raise AdressNotFoundError(f"No se ha encontrado ninguna dirección parecida a {direccion}")
    mejor_coincidencia = coincidencias[0]
    # Ahora buscamos esa cadena exacta en el dataframe y sacamos la primera fila
    coincidencia = callejero["DIRECCION_TEXTO"] == mejor_coincidencia
    fila = callejero[coincidencia].iloc[0]
    latitud = float(fila["LATITUD"])
    longitud = float(fila["LONGITUD"])
    return latitud, longitud


############## Parte 4 ##############


def carga_grafo() -> nx.MultiDiGraph:
    """ Función que recupera el quiver de calles de Madrid de OpenStreetMap.
    Args: None
    Returns:
        nx.MultiDiGraph: Quiver de las calles de Madrid.
    Raises:
        ServiceNotAvailableError: Si no es posible recuperar el grafo de OpenStreetMap.
    """
    fichero = "madrid.graphml"
    # Si existe el grafo no lo carga
    if os.path.exists(fichero):
        G = ox.load_graphml(fichero)
    else:
        try:
            G = ox.graph_from_place("Madrid, Spain", network_type="drive")
        except:
            raise ServiceNotAvailableError("No ha sido posible recuperar el grafo de OpenStreetMap")
        ox.save_graphml(G, fichero)
    return G

def procesa_grafo(multidigrafo:nx.MultiDiGraph) -> nx.DiGraph:
    """ Función que recupera el quiver de calles de Madrid de OpenStreetMap y lo limpia.
    Args:
        multidigrafo: multidigrafo de las calles de Madrid obtenido de OpenStreetMap.
    Returns:
        nx.DiGraph: Grafo dirigido y sin bucles asociado al multidigrafo dado y limpio.
    Raises: None
    """
    # Convierte de multidigrafo a digrafo
    G = ox.convert.to_digraph(multidigrafo)
    bucles = list(nx.selfloop_edges(G))
    G.remove_edges_from(bucles)
    # Recorremos las aristas
    for _, _, data in G.edges(data=True):
        highway = data.get("highway")
        # Si es una lista hay que arreglarlo
        if isinstance(highway, list) and highway:
            vel_ant = float("-inf")
            tipo = ""
            # Mira las velocidades máximas de cada uno de los tipos de vía
            for i in highway:
                vel = float(MAX_SPEEDS.get(i))
                # Selecciona la vía con la mayor velicidad máxima
                if vel > vel_ant:
                    vel_ant = vel
                    tipo = i
            # Cambia el tipo de la vía
            data["highway"] = tipo
        
        name = data.get("name")
        if isinstance(name, list):
            # Seleccionamos el primer nombre
            data["name"] = name[0]

        ms = data.get("maxspeed")
        # Si está vacía "maxspeed" la cambiamos según el diccionario
        if ms is None or ms == "" or ms == []:
            h = data.get("highway")
            data["maxspeed"] = float(MAX_SPEEDS[h])
        else:
            # Hay maxspeed pero es lista
            if isinstance(ms, list):
                valores = []
                # Pasamos valores a float
                for v in ms:
                    valores.append(float(v))
                # Cogemos el máximo
                data["maxspeed"] = max(valores)
            else:
                # Si es un número normal
                try:
                    data["maxspeed"] = float(data["maxspeed"])
                except:
                    h = data.get("highway")
                    data["maxspeed"] = float(MAX_SPEEDS.get(h))
    return G

def dibujar_grafo_nx(G:nx.DiGraph, min_long=0.0000001):
    xs, ys, us, vs = [], [], [], []

    for i, (u,v) in enumerate(G.edges):
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]

        dx = x2 - x1
        dy = y2 - y1

        norm = np.hypot(dx, dy)
        if norm == 0:
            continue

        if norm < min_long:
            factor = min_long / norm
            dx *= factor
            dy *= factor

        xs.append(x1)
        ys.append(y1)
        us.append(dx)
        vs.append(dy)

    plt.figure(figsize=(10,10))
    plt.quiver(
        xs, ys, us, vs,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        headwidth=4,
        headlength=6,
        color="black"
    )
    plt.axis("equal")
    plt.axis("off")
    plt.show()