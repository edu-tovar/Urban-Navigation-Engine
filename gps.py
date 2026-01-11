import callejero
from typing import Tuple,List
import grafo_pesado
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import math


def crear_grafo() -> Tuple[pd.DataFrame, nx.DiGraph]:

    df = callejero.carga_callejero()
    grafo = callejero.carga_grafo()
    grafo_p = callejero.procesa_grafo(grafo)

    return df, grafo_p

def encontrar_nodo(direccion:str, G: nx.DiGraph, df:pd.DataFrame)-> int:
    lat, lon = callejero.busca_direccion_fuzzy(direccion, df)
    
    nodo = ox.nearest_nodes(G, lon, lat)
    return nodo

def _nombre_calle(G: nx.DiGraph, u: int, v: int) -> str:
    """Devuelve un nombre de calle para la arista (u, v)."""
    datos = G[u][v]
    nombre = datos.get("name")
    if nombre is None or str(nombre).strip() == "":
        return "vía sin nombre"

    return str(nombre)

def _calcular_giro(camino: List[int], G: nx.DiGraph, idx_inicio_segmento: int) -> str | None:
    """Calcula el tipo de giro en el nodo donde empieza un nuevo tramo. Se usa el nodo anterior (n0), el nodo de cruce (n1) y el siguiente (n2)
    """
    # Necesitamos tener un nodo anterior y uno siguiente, no puede ser ni el primero ni el último
    if idx_inicio_segmento <= 0 or idx_inicio_segmento + 1 >= len(camino):
        return None

    n0 = camino[idx_inicio_segmento - 1]
    n1 = camino[idx_inicio_segmento]
    n2 = camino[idx_inicio_segmento + 1]

    x0, y0 = G.nodes[n0]["x"], G.nodes[n0]["y"]
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
    # vector de llegada al cruce
    v1x, v1y = x1 - x0, y1 - y0
    # vector de salida del cruce
    v2x, v2y = x2 - x1, y2 - y1

    # Comprobamos que no sean vectores nulos
    if v1x == 0 and v1y == 0:
        return None
    if v2x == 0 and v2y == 0:
        return None

    # Hacemos el producto cruzado para saber el signo del giro
    cross = v1x * v2y - v1y * v2x
    # Hacemos el producto escalar para saber el signo del giro
    dot = v1x * v2x + v1y * v2y

    eps = 0.000001

    # Si el giro es prácticamente recto
    if abs(cross) < eps and dot > 0:
        # seguir recto -> no hace falta decir 'gira'
        return None

    if cross > 0:
        return "gira a la izquierda"
    else:
        return "gira a la derecha"

def _frase_segmento(nombre: str, dist_m: float, es_primero: bool, giro: str | None) -> str:
    """Construye una frase de navegación para un tramo de una calle."""
    dist_km = dist_m / 1000.0

    if es_primero:
        inicio = "Sal desde el origen y "
    else:
        if giro:
            inicio = f"Luego {giro} y "
        else:
            inicio = "Luego "

    if nombre == "vía sin nombre":
        return f"{inicio}continúa durante {dist_km:.2f} km por una vía sin nombre."
    else:
        return f"{inicio}sigue por {nombre} durante {dist_km:.2f} km."

# def construir_instrucciones(camino:List[int], G:nx.DiGraph) -> List[str]:
#     """A partir de una lista de nodos del grafo, construye instrucciones de navegación."""
#     if not camino or len(camino) == 1:
#         return ["Origen y destino coinciden. Ya estás en tu destino."]

#     instrucciones = []
#     total_dist = 0.0

#     # Inicializa con la primera arista
#     u = camino[0]
#     v = camino[1]
#     nombre_actual = _nombre_calle(G, u, v)
#     dist_actual = float(G[u][v].get("length", 0.0))

#     # Se recorren el resto de aristas
#     for i in range(1, len(camino) - 1):
#         u = camino[i]
#         v = camino[i + 1]
#         datos = G[u][v]
#         nombre = _nombre_calle(G, u, v)
#         longitud = float(datos.get("length", 0.0))

#         if nombre == nombre_actual:
#             # Si es el mismo nombre de calle -> acumulamos distancia
#             dist_actual += longitud
#         else:
#             # Si no, cambiamos de calle
#             instrucciones.append(
#                 _frase_segmento(nombre_actual, dist_actual, es_primero=(len(instrucciones) == 0))
#             )
#             total_dist += dist_actual

#             # Empezamos nuevo tramo
#             nombre_actual = nombre
#             dist_actual = longitud

#     # Añadimos el último tramo
#     instrucciones.append(
#         _frase_segmento(nombre_actual, dist_actual, es_primero=(len(instrucciones) == 0)))
#     total_dist += dist_actual
#     instrucciones.append(f"Distancia total aproximada: {total_dist/1000:.2f} km.")
#     return instrucciones
def construir_instrucciones(camino: List[int], G: nx.DiGraph) -> List[str]:
    """A partir de una lista de nodos del grafo, construye instrucciones de navegación,
    indicando izquierda/derecha cuando se cambia de calle.
    """
    if not camino or len(camino) == 1:
        return ["Origen y destino coinciden. Ya estás en tu destino."]

    # Primero construimos segmentos de la calle
    segmentos = []

    # Inicializamos con la primera arista
    u0 = camino[0]
    v0 = camino[1]
    nombre_actual = _nombre_calle(G, u0, v0)
    dist_actual = float(G[u0][v0].get("length", 0.0))
    # índice en 'camino' donde empieza este segmento
    inicio_idx = 0

    # Recorremos el resto de aristas
    for i in range(1, len(camino) - 1):
        u = camino[i]
        v = camino[i + 1]
        datos = G[u][v]
        nombre = _nombre_calle(G, u, v)
        longitud = float(datos.get("length", 0.0))
        if nombre == nombre_actual:
            # Misma calle -> acumulamos distancia
            dist_actual += longitud
        else:
            # Cerramos el segmento anterior: desde 'inicio_idx' hasta 'i'
            segmentos.append(
                {
                    "nombre": nombre_actual,
                    "dist": dist_actual,
                    "inicio": inicio_idx,
                    "fin": i,
                }
            )
            # Nuevo segmento
            nombre_actual = nombre
            dist_actual = longitud
            inicio_idx = i
    # Añadimos el último segmento (hasta el último nodo)
    segmentos.append(
        {
            "nombre": nombre_actual,
            "dist": dist_actual,
            "inicio": inicio_idx,
            "fin": len(camino) - 1,
        }
    )
    # Construimos las frases usando los segmentos y calculando giros
    instrucciones: List[str] = []
    total_dist = 0.0

    for j, seg in enumerate(segmentos):
        nombre = seg["nombre"]
        dist = seg["dist"]

        if j == 0:
            giro = None
        else:
            # El giro se calcula en el nodo donde empieza este segmento
            idx_inicio = seg["inicio"]
            giro = _calcular_giro(camino, G, idx_inicio)
        frase = _frase_segmento(nombre=nombre, dist_m=dist, es_primero=(j == 0), giro=giro,)
        instrucciones.append(frase)
        total_dist += dist
    instrucciones.append(f"Distancia total aproximada: {total_dist/1000:.2f} km.")
    return instrucciones

def mostrar_camino(camino:List[int], grafo:nx.DiGraph) -> None:
    """Muestra por pantalla la lista de instrucciones para un camino"""
    instrucciones = construir_instrucciones(camino, grafo)

    print("\nInstrucciones de ruta:")
    for i, instruccion in enumerate(instrucciones, start=1):
        print(f"{i}. {instruccion}")

def dibujar_ruta(camino:List[int], grafo:nx.DiGraph) -> None:
    """Dibuja el grafo completo y resalta la ruta indicada"""
    if not camino:
        print("No hay ruta que dibujar (el camino está vacío).")
        return
    pos = {n:(datos["x"], datos["y"]) for n, datos in grafo.nodes(data=True)}
    path_edges = list(zip(camino[:-1], camino[1:]))
    plt.figure(figsize=(10, 10))

    # Todas las calles en gris
    nx.draw_networkx_edges(
        grafo,
        pos,
        edge_color="lightgray",
        width=0.5,
        arrows=False,
    )
    # Ruta en rojo, un pelín más ancha
    nx.draw_networkx_edges(
        grafo,
        pos,
        edgelist=path_edges,
        edge_color="red",
        width=1.0,
        arrows=False,
    )
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def bucle_interactivo(df:pd.DataFrame, grafo: nx.DiGraph):
    origen = input("Introduzca la direccion de la que se quiere partir.")
    destino = input("Introduzca la direccion de destino.")

    while origen and destino:
        opcion = 0
        nodo_or = encontrar_nodo(origen, grafo, df)
        nodo_de = encontrar_nodo(destino, grafo, df)

        opciones = ["1", "2", "3"]
        while opcion not in opciones:
            opcion = input("""Escoja una opcion:
                                1- Ruta mas corta.
                                2- Ruta mas rapida.
                                3- Ruta mas rapida teniendo en cuenta semaforos.""")
            

        if opcion == "1":
            funcion_peso = grafo_pesado.mas_corto

        elif opcion == "2":
            funcion_peso = grafo_pesado.mas_rapido
        else:
            funcion_peso = grafo_pesado.mas_rapido_semaforos

        camino = grafo_pesado.camino_minimo(grafo, funcion_peso, nodo_or, nodo_de)
        if not camino:
            print("No existe ruta entre las direcciones indicadas.")
        else:
            mostrar_camino(camino, grafo)
            dibujar_ruta(camino, grafo)
        origen = input("Introduzca la direccion de la que se quiere partir.")
        destino = input("Introduzca la direccion de destino.")


def main():
    df, grafo = crear_grafo()
    bucle_interactivo(df, grafo)


if __name__ == '__main__':
    main()