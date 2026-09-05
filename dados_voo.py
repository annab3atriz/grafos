import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
import json

# O parâmetro low_memory=False tira aquele aviso do terminal
df = pd.read_csv('VRA_20267.csv', sep=';', skiprows=1, low_memory=False)

df = df[df['Situação Voo'] == 'REALIZADO']
df = df.dropna(subset=['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino'])
df = df[df['ICAO Aeródromo Origem'] != df['ICAO Aeródromo Destino']]

# Cria uma tupla com os prefixos ICAO do Brasil
prefixos_br = ('SB', 'SD', 'SI', 'SJ', 'SN', 'SS', 'SW')

# Filtra origem e destino para garantir que ambos são do Brasil
df = df[df['ICAO Aeródromo Origem'].str.startswith(prefixos_br)]
df = df[df['ICAO Aeródromo Destino'].str.startswith(prefixos_br)]

arestas = df.groupby(['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino']).size().reset_index(name='peso')

print(arestas.head(20))
print(len(arestas))

# Cria a rede direcionada
G = nx.from_pandas_edgelist(
    arestas,
    source='ICAO Aeródromo Origem',
    target='ICAO Aeródromo Destino',
    edge_attr='peso',
    create_using=nx.DiGraph()
)

# Confirma a criação imprimindo os dados básicos do grafo
print(G)

# 1. Carrega as coordenadas (pula a linha de data e usa latin1 para os acentos)
df_coords = pd.read_csv('AerodromosPublicos.csv', sep=';', encoding='latin1', skiprows=1)

# 2. Cria o dicionário de posições: { 'SBRJ': (Longitude, Latitude) }
posicoes = {}
for no in G.nodes():
    # Busca a linha do aeroporto no dataframe
    aeroporto = df_coords[df_coords['Código OACI'] == no]
    
    # Se encontrar as coordenadas, adiciona ao dicionário
    if not aeroporto.empty:
        lon = aeroporto.iloc[0]['LONGEOPOINT']
        lat = aeroporto.iloc[0]['LATGEOPOINT']
        posicoes[no] = (lon, lat)


nos_sem_coord = [no for no in G.nodes() if no not in posicoes]
G.remove_nodes_from(nos_sem_coord)

# ==========================================
# 1. CARACTERÍSTICAS ESTRUTURAIS
# ==========================================
print("\n--- CARACTERÍSTICAS ESTRUTURAIS ---")
print(f"Número de Nós (Aeroportos): {G.number_of_nodes()}")
print(f"Número de Arestas (Rotas): {G.number_of_edges()}")
print(f"Densidade da Rede: {nx.density(G):.4f}")

# Para clustering e caminhos curtos, usamos a versão não-direcionada
G_undirected = G.to_undirected()
print(f"Coeficiente de Aglomeração (Clustering): {nx.average_clustering(G_undirected):.4f}")

# Extrai o maior componente conectado (para calcular distância e diâmetro sem erros)
maior_componente = max(nx.connected_components(G_undirected), key=len)
G_core = G_undirected.subgraph(maior_componente)

print(f"Diâmetro da Rede Principal: {nx.diameter(G_core)}")
print(f"Distância Média entre Aeroportos: {nx.average_shortest_path_length(G_core):.4f} saltos")


# ==========================================
# 2. MÉTRICAS DE CENTRALIDADE
# ==========================================
print("\n--- TOP 5 AEROPORTOS POR CENTRALIDADE ---")

# Função auxiliar para pegar os top 5 de um dicionário de notas
def exibir_top5(nome_metrica, dict_centralidade):
    # Ordena o dicionário do maior para o menor
    top5 = sorted(dict_centralidade.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n{nome_metrica}:")
    for aeroporto, nota in top5:
        print(f" - {aeroporto}: {nota:.4f}")

# Calcula as métricas usando o grafo direcionado G original
exibir_top5("Grau de Entrada (Voos que chegam)", nx.in_degree_centrality(G))
exibir_top5("Grau de Saída (Voos que partem)", nx.out_degree_centrality(G))
exibir_top5("Closeness Centrality (Proximidade / Fácil acesso)", nx.closeness_centrality(G))
exibir_top5("Betweenness Centrality (Intermediação / Conexões)", nx.betweenness_centrality(G))
exibir_top5("PageRank (Importância Global)", nx.pagerank(G))

# Eigenvector costuma dar erro de convergência em grafos direcionados esparsos,
# usar a versão não-direcionada resolve e mantém a lógica matemática.
exibir_top5("Eigenvector Centrality (Influência da Vizinhança)", nx.eigenvector_centrality(G_undirected, max_iter=1000))





# Cria a figura
plt.figure(figsize=(12, 12))
ax = plt.gca()
ax.set_facecolor('#f4f4f4') # Fundo cinza claro para o oceano

# 1. Baixa as coordenadas dos estados do Brasil
url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
resposta = urllib.request.urlopen(url)
mapa_brasil = json.loads(resposta.read())

# 2. Desenha o mapa estado por estado
for estado in mapa_brasil['features']:
    geometria = estado['geometry']
    
    if geometria['type'] == 'Polygon':
        poligonos = [geometria['coordinates'][0]]
    else: # MultiPolygon (estados com ilhas)
        poligonos = [poly[0] for poly in geometria['coordinates']]
        
    for coords in poligonos:
        xs, ys = zip(*coords)
        # Preenche de branco e faz a borda cinza
        ax.fill(xs, ys, color='white', edgecolor='darkgray', linewidth=1, zorder=1)

# 3. Desenha a rede de aeroportos por cima do mapa
nx.draw_networkx_nodes(G, pos=posicoes, ax=ax, node_size=11, node_color='red', alpha=0.8)
nx.draw_networkx_edges(G, pos=posicoes, ax=ax, edge_color='black', alpha=0.2, width=0.4, arrowsize=6)

plt.title("Rede de Aeroportos no Brasil")
# Ajusta a "câmera" para os limites exatos do país
ax.set_xlim(-75, -34)
ax.set_ylim(-35, 6)

# Remove os números das bordas
ax.set_xticks([])
ax.set_yticks([])

plt.show()
