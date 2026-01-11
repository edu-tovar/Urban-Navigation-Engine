"""
grafo.py

Matemática Discreta - IMAT
ICAI, Universidad Pontificia Comillas

Grupo: GP08B
Integrantes:
    - Alejandro Royo-Villanova Seguí
    - Eduardo Tovar Ruíz

Descripción:
Librería para el análisis de grafos pesados.
"""

from typing import List,Tuple,Dict,Callable,Union
import networkx as nx
import sys
import callejero
import itertools
import heapq #Librería para la creación de colas de prioridad

INFTY=sys.float_info.max #Distincia "infinita" entre nodos de un grafo

"""
En las siguientes funciones, las funciones de peso son funciones que reciben un grafo o digrafo y dos vértices y devuelven un real (su peso)
Por ejemplo, si las aristas del grafo contienen en sus datos un campo llamado 'valor', una posible función de peso sería:

def mi_peso(G:nx.Graph,u:object, v:object):
    return G[u][v]['valor']

y, en tal caso, para calcular Dijkstra con dicho parámetro haríamos

camino=dijkstra(G,mi_peso,origen, destino)


"""

def mas_corto(G:Union[nx.Graph, nx.DiGraph], u:object, v:object) -> float:
    """Función de peso para calcular la ruta más corta en metros. Devuelve la longitud de la arista (u, v) en metros, usando el atributo 'length'

    Args:
        G: grafo de calles
        u: nodo origen de la arista
        v: nodo destino de la arista

    Returns:
        float: longitud de la arista (u, v) en metros
    """
    return float(G[u][v]["length"])

def _velocidad_kmh(G:nx.Graph, u:object, v:object) -> float:
    """Devuelve la velocidad máxima (en km/h) para la arista (u, v).
    """
    datos = G[u][v]
    velocidad_kmh = None

    # Intenta usar 'maxspeed' de la arista
    maxspeed = datos.get("maxspeed")
    if maxspeed is not None:
        texto = str(maxspeed).strip()
        if texto:
            try:
                velocidad_kmh = float(texto)
            except ValueError:
                velocidad_kmh = None

    # Si no hay maxspeed usable, usar tipo de vía 'highway'
    if velocidad_kmh is None:
        highway = datos.get("highway")
        valor = callejero.MAX_SPEEDS.get(highway)
        velocidad_kmh = float(valor)
    return velocidad_kmh

def mas_rapido(G:nx.Graph, u:object, v:object) -> float:
    """Función de peso para calcular la ruta más rápida.
    El peso es el tiempo de viaje estimado en segundos, suponiendo que se circula a la velocidad máxima permitida en esa vía
    El tiempo será: tiempo = longitud (m) / velocidad (m/s)

    Args:
        G: grafo de calles
        u: nodo origen de la arista
        v: nodo destino de la arista

    Returns:
        float: tiempo de recorrido de la arista (u, v) en segundos
    """
    datos = G[u][v]
    longitud = datos.get("length")
    longitud_m = float(longitud)
    vel_kmh = _velocidad_kmh(G, u, v)
    # Paso la velocidad a m/s
    vel_m_s = vel_kmh / 3.6
    # Calculo el tiempo
    tiempo_seg = longitud_m / vel_m_s
    return tiempo_seg

def mas_rapido_semaforos(G:nx.Graph, u:object, v:object) -> float:
    """Función de peso para calcuar la ruta más rápida considerando semáforos.
    El peso es el tiempo de viaje esperado en segundos, suponiendo que se circula a la velocidad máxima permitida en esa vía, y que
    cada vez que se pasa por un cruce hay probabilidad p = 0.8 de tener que parar 30 s. Esto equivale a añadir un tiempo esperado de 24 s por arista

    Args:
        G: grafo de calles
        u: nodo origen de la arista
        v: nodo destino de la arista

    Returns:
        float: Tiempo esperado de recorrido de la arista (u, v) en segundos.
    """
    # Tiempo base de circular a velocidad máxima
    tiempo_base = mas_rapido(G, u, v)
    # Probabilidad de tener que parar
    p = 0.8
    tiempo_parada = 30.0
    espera_esperada = p * tiempo_parada
    return tiempo_base + espera_esperada


def dijkstra(G:Union[nx.Graph, nx.DiGraph], peso:Union[Callable[[nx.Graph,object,object],float], Callable[[nx.DiGraph,object,object],float]], origen:object)-> Dict[object,object]:
    """ Calcula un Árbol de Caminos Mínimos para el grafo pesado partiendo
    del vértice "origen" usando el algoritmo de Dijkstra. Calcula únicamente
    el árbol de la componente conexa que contiene a "origen".
    
    Args:
        origen (object): vértice del grafo de origen
    Returns:
        Dict[object,object]: Devuelve un diccionario que indica, para cada vértice alcanzable
            desde "origen", qué vértice es su padre en el árbol de caminos mínimos.
    Raises:
        TypeError: Si origen no es "hashable".
    Example:
        Si G.dijksra(1)={2:1, 3:2, 4:1} entonces 1 es padre de 2 y de 4 y 2 es padre de 3.
        En particular, un camino mínimo desde 1 hasta 3 sería 1->2->3.
    """
    
    if not isinstance(origen, (int, str, tuple, frozenset)) and not hasattr(origen, "__hash__"):
        raise TypeError("El vértice origen debe ser hashable.")

    padre = {}
    dist = {}
    visitado = {}

    for v in G.nodes:
        padre[v] = None
        visitado[v] = False
        dist[v] = INFTY

    dist[origen] = 0

    contador = itertools.count()
    Q = [(0, next(contador), origen)]

    while Q:
        dist_v, _, v = heapq.heappop(Q)

        if not visitado[v]:
            visitado[v] = True

            for x in G.neighbors(v):
                w_vx = peso(G, v, x)
                if dist[x] > dist[v] + w_vx:
                    dist[x] = dist[v] + w_vx
                    padre[x] = v
                    heapq.heappush(Q, (dist[x], next(contador), x))

    if origen in padre:
        del padre[origen]

    return padre


def camino_minimo(G:Union[nx.Graph, nx.DiGraph], peso:Union[Callable[[nx.Graph,object,object],float], Callable[[nx.DiGraph,object,object],float]] ,origen:object,destino:object)->List[object]:
    """ Calcula el camino mínimo desde el vértice origen hasta el vértice
    destino utilizando el algoritmo de Dijkstra.
    
    Args:
        G (nx.Graph o nx.Digraph): grafo a grado dirigido
        peso (función): función que recibe un grafo o grafo dirigido y dos vértices del mismo y devuelve el peso de la arista que los conecta
        origen (object): vértice del grafo de origen
        destino (object): vértice del grafo de destino
    Returns:
        List[object]: Devuelve una lista con los vértices del grafo por los que pasa
            el camino más corto entre el origen y el destino. El primer elemento de
            la lista es origen y el último destino.
    Example:
        Si dijksra(G,peso,1,4)=[1,5,2,4] entonces el camino más corto en G entre 1 y 4 es 1->5->2->4.
    Raises:
        TypeError: Si origen o destino no son "hashable".
    """

    if not isinstance(origen, (int, str, tuple, frozenset)) and not hasattr(origen, "__hash__"):
        raise TypeError("El vértice origen debe ser hashable.")
    
    if not isinstance(destino, (int, str, tuple, frozenset)) and not hasattr(destino, "__hash__"):
        raise TypeError("El vértice destino debe ser hashable.")
    
    if origen == destino:
        return [origen]
    camino = [destino]
    padres = dijkstra(G, peso, origen)

    if destino not in padres:
        return []


    actual = destino
    while actual != origen:
        actual = padres[actual]
        camino.append(actual)

    camino.reverse()

    return camino


def prim(G:nx.Graph, peso:Callable[[nx.Graph,object,object],float])-> Dict[object,object]:
    """ Calcula un Árbol Abarcador Mínimo para el grafo pesado
    usando el algoritmo de Prim.
    
    Args: None
    Returns:
        G (nx.Graph): grafo
        peso (función): función que recibe un grafo y dos vértices del grafo y devuelve el peso de la arista que los conecta
        Dict[object,object]: Devuelve un diccionario que indica, para cada vértice del
            grafo, qué vértice es su padre en el árbol abarcador mínimo.
    Raises: None
    Example:
        Si prim(G,peso)={1: None, 2:1, 3:2, 4:1} entonces en un árbol abarcador mínimo tenemos que:
            1 es una raíz (no tiene padre)
            1 es padre de 2 y de 4
            2 es padre de 3
    """
    contador = itertools.count()

    padre = {}
    coste = {}
    Q = []

    for v in G.nodes:
        padre[v] = None
        coste[v] = INFTY
        heapq.heappush(Q, (coste[v], next(contador), v))

    en_Q = set(G.nodes)

    while Q:
        coste_v, _, v = heapq.heappop(Q)

        if v not in en_Q:
            continue

        en_Q.remove(v)

        for x in G.neighbors(v):
            if x in en_Q:
                w_vx = peso(G, v, x)
                if w_vx < coste[x]:
                    coste[x] = w_vx
                    padre[x] = v
                    heapq.heappush(Q, (coste[x], next(contador), x))

    return padre


def kruskal(G:nx.Graph, peso:Callable[[nx.Graph,object,object],float])-> List[Tuple[object,object]]:
    """ Calcula un Árbol Abarcador Mínimo para el grafo
    usando el algoritmo de Kruskal.
    
    Args:
        G (nx.Graph): grafo
        peso (función): función que recibe un grafo y dos vértices del grafo y devuelve el peso de la arista que los conecta
    Returns:
        List[Tuple[object,object]]: Devuelve una lista [(s1,t1),(s2,t2),...,(sn,tn)]
            de los pares de vértices del grafo que forman las aristas
            del arbol abarcador mínimo.
    Raises: None
    Example:
        En el ejemplo anterior en que prim(G,peso)={1:None, 2:1, 3:2, 4:1} podríamos tener, por ejemplo,
        kruskal(G,peso)=[(1,2),(1,4),(3,2)]
    """
    lista = []
    for u, v in G.edges:
        lista.append((peso(G,u,v),u,v))
    
    lista.sort(key=lambda x:x[0])

    C = {v: {v} for v in G.nodes}

    aristas_aam = []

    while lista:
        c, u, v = lista.pop(0)

        if C[u] != C[v]:
            aristas_aam.append((u,v))

            comp_u = C[u]
            comp_v = C[v]
            nueva_comp = comp_u.union(comp_v)

            for w in nueva_comp:
                C[w] = nueva_comp

    return aristas_aam