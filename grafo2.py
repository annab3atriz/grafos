# Anna Beatriz Souza e Silva - 13694103
# Milena Rodrigues Monteiro - 12566157

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import plotly.express as px
import collections
import urllib.request
import json

# LEITURA DOS DADOS
df_vra = pd.read_csv('VRA_20267.csv', sep=';', skiprows=1, low_memory=False)
df_aerodromos = pd.read_csv('AerodromosPublicos.csv', sep=';', skiprows=1, low_memory=False, encoding='ISO-8859-1')


# LIMPEZA VRA (Voos)
df_vra_filtrado = (
    df_vra
    .drop(columns=['Código Justificativa']) 
    .loc[ 
        lambda df: ( 
            (df['Situação Voo'] != 'CANCELADO') & 
            (df['Código Autorização (DI)'].isin(['0', '4', 'C'])) & 
            (df['Código Tipo Linha'].isin(['N', 'C']))
        )
    ]
)

# LIMPEZA AERÓDROMOS
df_aerodromos_filtrado = df_aerodromos.drop(columns=['Portaria de Registro', 'Link Portaria']) 
df_aerodromos_filtrado = df_aerodromos_filtrado[df_aerodromos_filtrado['Situação'] != 'Interditado'] 
df_aerodromos_filtrado['Validade do Registro'] = pd.to_datetime(df_aerodromos_filtrado['Validade do Registro'], format='%d/%m/%Y', errors='coerce')

comparison_date = pd.to_datetime('01/07/2026', format='%d/%m/%Y') 
df_aerodromos_filtrado = df_aerodromos_filtrado[
    df_aerodromos_filtrado['Validade do Registro'].isna() | 
    (df_aerodromos_filtrado['Validade do Registro'] > comparison_date) 
]

# CONSTRUÇÃO DA REDE PURA (Sem coordenadas)
df_voos = df_vra_filtrado.dropna(subset=['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino']).copy() 
df_voos = df_voos[df_voos['ICAO Aeródromo Origem'] != df_voos['ICAO Aeródromo Destino']] 

# CRIA ARESTAS COM PESO (número de voos)
arestas_calculo = df_voos.groupby(['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino']).size().reset_index(name='peso') 

# RREPRESENTAÇÃO COMPUTACIONAL
G = nx.from_pandas_edgelist(
    arestas_calculo,
    source='ICAO Aeródromo Origem',
    target='ICAO Aeródromo Destino',
    edge_attr='peso',
    create_using=nx.DiGraph()
)

# CARACTERIZAÇÃO ESTRUTURAL

print(f"Número de Nós (Aeroportos): {G.number_of_nodes()}")
print(f"Número de Arestas (Rotas): {G.number_of_edges()}")
print(f"Densidade da Rede: {nx.density(G):.4f}")

graus = [grau for no, grau in G.degree()]
print(f"Grau Médio da Rede: {(sum(graus) / len(graus)):.2f} conexões por aeroporto")

G_undirected = G.to_undirected()
print(f"Coeficiente de Aglomeração (Clustering): {nx.average_clustering(G_undirected):.4f}")
print(f"Número de Componentes Conectados: {nx.number_connected_components(G_undirected)}")


# INVESTIGAÇÃO DOS COMPONENTES CONEXOS
componentes = list(nx.connected_components(G_undirected))

# MOSTRA O TAMANHO E OS AEROPORTOS DE CADA COMPONENTE
for i, comp in enumerate(componentes):
    print(f"Componente {i+1} tem {len(comp)} aeroportos:")
    print(comp)
    print("-" * 30)

maior_componente = max(nx.connected_components(G_undirected), key=len)
G_core = G_undirected.subgraph(maior_componente)
print(f"Diâmetro da Rede Principal: {nx.diameter(G_core)}")
print(f"Distância Média entre Aeroportos: {nx.average_shortest_path_length(G_core):.4f} saltos")


# MÉTRICAS DE CENTRALIDADE

print("\n--- TOP 5 AEROPORTOS POR CENTRALIDADE ---")

# FUNÇÃO AUXILIAR PARA PEGAR OS TOP 5 DE UM DICIONÁRIO
def exibir_top5(nome_metrica, dict_centralidade):
    # Ordena o dicionário do maior para o menor
    top5 = sorted(dict_centralidade.items(), key=lambda x: x[1], reverse=True)[:5] 
    print(f"\n{nome_metrica}:")
    for aeroporto, nota in top5:
        print(f" - {aeroporto}: {nota:.4f}")

# CALCULA AS MÉTRICAS USANDO O GRAFO DIRECIONADO G ORIGINAL
exibir_top5("Grau de Entrada (Voos que chegam)", dict(G.in_degree()))  
exibir_top5("Grau de Saída (Voos que partem)", dict(G.out_degree()))  
exibir_top5("In-Closeness Centrality (Proximidade / Fácil acesso)", nx.closeness_centrality(G)) 
exibir_top5("Out-Closeness Centrality (Proximidade / Fácil acesso)", nx.closeness_centrality(G.reverse()))
exibir_top5("Betweenness Centrality (Intermediação / Conexões)", nx.betweenness_centrality(G))
exibir_top5("PageRank (Importância Global)", nx.pagerank(G))

# Eigenvector costuma dar erro de convergência em grafos direcionados por isso usamos a versão não direcionada do nossa rede
exibir_top5("Eigenvector Centrality (Influência da Vizinhança)", nx.eigenvector_centrality(G_undirected, max_iter=1000))


# REDES CLÁSSICAS PARA COMPARAÇÃO
nos = G.number_of_nodes();
arestas = G.number_of_edges();

# REDE ALEATÓRIA (ERDŐS-RÉNYI)
# Podemos passar o número de nós e arestas diretamente
G_random = nx.gnm_random_graph(n=nos, m=arestas, directed=True)


# REDE PEQUENO MUNDO (WATTS-STROGATZ)
# O parâmetro 'k' representa os vizinhos iniciais (grau médio).
# Grau médio (não direcionado) = (2 * arestas) / nos
k_medio = round((2 * arestas) / nos) 
k_medio = max(2, k_medio)
G_small_world = nx.watts_strogatz_graph(n=nos, k=k_medio, p=0.1)


# 3. Scale-Free (Barabási-Albert)
# O parâmetro 'm' é o número de arestas que cada novo nó cria.
# m equivale a aproximadamente metade do grau médio (arestas / nos).
m_ba = round(arestas / nos)
m_ba = max(1, min(m_ba, nos - 1))
G_scale_free = nx.barabasi_albert_graph(n=nos, m=m_ba)

# MEDIDAS DA REDE ORIGINAL
G_undirected = G.to_undirected()  # Converte para não-direcionado para cálculo de clustering
clust_orig = nx.average_clustering(G_undirected)
# Usa connected_components para garantir a existência de caminhos no cálculo de path length
comp_orig = max(nx.connected_components(G_undirected), key=len) 
path_orig = nx.average_shortest_path_length(G_undirected.subgraph(comp_orig))

graus_orig = [grau for no, grau in G.in_degree()] 
cont_orig = collections.Counter(graus_orig)


# MEDIDAS DA REDE ALEATÓRIA
clust_rand = nx.average_clustering(G_random)
comp_rand = max(nx.strongly_connected_components(G_random), key=len)
path_rand = nx.average_shortest_path_length(G_random.subgraph(comp_rand))

graus_rand = [grau for no, grau in G_random.in_degree()] 
cont_rand = collections.Counter(graus_rand)

# MEDIDAS DA REDE DE PEQUENO MUNDO
clust_sw = nx.average_clustering(G_small_world)
comp_sw = max(nx.connected_components(G_small_world), key=len)
path_sw = nx.average_shortest_path_length(G_small_world.subgraph(comp_sw))

graus_sw = [grau for no, grau in G_small_world.degree()]
cont_sw = collections.Counter(graus_sw)

# MEDIDAS DA REDE LIVRE DE CONTEXTO
clust_sf = nx.average_clustering(G_scale_free)
comp_sf = max(nx.connected_components(G_scale_free), key=len)
path_sf = nx.average_shortest_path_length(G_scale_free.subgraph(comp_sf))

graus_sf = [grau for no, grau in G_scale_free.degree()]
cont_sf = collections.Counter(graus_sf)

# EXIBE DOS RESULTADOS
print("Métricas calculadas sobre o maior componente (fortemente conexo para direcionadas):")
print(f"Original    - Clustering: {clust_orig:.4f} | Path Length: {path_orig:.4f}")
print(f"Aleatória   - Clustering: {clust_rand:.4f} | Path Length: {path_rand:.4f}")
print(f"Small-World - Clustering: {clust_sw:.4f} | Path Length: {path_sw:.4f}")
print(f"Scale-Free  - Clustering: {clust_sf:.4f} | Path Length: {path_sf:.4f}")

# GRÁFICO DE DISTRIBUIÇÃO DE GRAU
plt.figure(figsize=(10, 6))

def plot_ccdf(graus, label):
    # Conta a frequência de cada grau
    contagem = collections.Counter(graus)
    graus_unicos = sorted(contagem.keys())
    
    # Calcula P(X >= k) apenas para os graus únicos
    total_nos = len(graus)
    ccdf = [sum(v for k, v in contagem.items() if k >= grau) / total_nos for grau in graus_unicos]
    
    plt.plot(graus_unicos, ccdf, marker='o', linestyle='-', label=label, alpha=0.7)

plot_ccdf(graus_orig, "Original")
plot_ccdf(graus_rand, "Aleatória")
plot_ccdf(graus_sw, "Small-World")
plot_ccdf(graus_sf, "Scale-Free")

plt.title("Distribuição de Grau Cumulativa (CCDF)")
plt.xlabel("Grau (k)")
plt.ylabel("P(X >= k)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# MAPAS

# JUNÇÕES DE TABELAS PARA CRIAÇÃO DOS MAPAS
aero_cols = ['Código OACI', 'CIAD', 'Nome', 'Município', 'UF', 'LATGEOPOINT', 'LONGEOPOINT']
df_aero = df_aerodromos_filtrado[aero_cols].copy()

# RENOMEIA ORIGEM
df_aero_origem = df_aero.rename(columns={
    'Código OACI': 'Código OACI Origem', 'CIAD': 'CIAD Origem', 'Nome': 'Nome Origem',
    'Município': 'Município Origem', 'UF': 'UF Origem', 
    'LATGEOPOINT': 'Latitude Origem', 'LONGEOPOINT': 'Longitude Origem'
})

# MERGE ORIGEM
df_vra_join = df_vra_filtrado.merge(
    df_aero_origem, left_on='ICAO Aeródromo Origem', right_on='Código OACI Origem', how='left', validate='many_to_one'
)

# RENOMEIA DESTINO
df_aero_destino = df_aero.rename(columns={
    'Código OACI': 'Código OACI Destino', 'CIAD': 'CIAD Destino', 'Nome': 'Nome Destino',
    'Município': 'Município Destino', 'UF': 'UF Destino', 
    'LATGEOPOINT': 'Latitude Destino', 'LONGEOPOINT': 'Longitude Destino'
})

# MERGE DESTINO
df_vra_join = df_vra_join.merge(
    df_aero_destino, left_on='ICAO Aeródromo Destino', right_on='Código OACI Destino', how='left', validate='many_to_one'
)

df_vra_join['UF Origem'] = df_vra_join['CIAD Origem'].str[:2]
df_vra_join['UF Destino'] = df_vra_join['CIAD Destino'].str[:2]


# GRÁFICO DE BARRAS: Top 15 Aeroportos em Quantidade de Voos (para comparação com aeroportos mais centrais)
movimentacao = pd.concat([
    df_vra_join[['ICAO Aeródromo Origem', 'Nome Origem', 'Município Origem', 'UF Origem']].rename(columns={
        'ICAO Aeródromo Origem': 'OACI', 'Nome Origem': 'Nome',
        'Município Origem': 'Município', 'UF Origem': 'UF'
    }),
    df_vra_join[['ICAO Aeródromo Destino', 'Nome Destino', 'Município Destino', 'UF Destino']].rename(columns={
        'ICAO Aeródromo Destino': 'OACI', 'Nome Destino': 'Nome',
        'Município Destino': 'Município', 'UF Destino': 'UF'
    })
], ignore_index=True)

movimentacao['Nome'] = movimentacao['Nome'].replace(r'^\s*$', pd.NA, regex=True)
movimentacao['Município'] = movimentacao['Município'].replace(r'^\s*$', pd.NA, regex=True)
movimentacao['UF'] = movimentacao['UF'].replace(r'^\s*$', pd.NA, regex=True)
movimentacao['OACI'] = movimentacao['OACI'].astype(str).str.strip().str.upper()
uf_fallback = movimentacao['OACI'].str.startswith(('SB', 'SD', 'SI', 'SJ', 'SN', 'SS', 'SW')).map({
    True: 'UF não informada', False: 'Exterior'
})
uf = movimentacao['UF'].fillna(uf_fallback).astype(str).str.strip().str.upper()
movimentacao['Aeroporto'] = (
    movimentacao['Nome']
    .fillna(movimentacao['Município'])
    .fillna(movimentacao['OACI'])
    .astype(str)
    .str.strip()
    .str.title()
    + ' ('
    + uf
    + ')'
)
top_aeroportos = movimentacao['Aeroporto'].value_counts().head(15).sort_values()

fig, ax = plt.subplots(figsize=(14, 8))
top_aeroportos.plot(kind='barh', ax=ax)
ax.bar_label(ax.containers[0], fmt='%d', padding=3)
ax.set_xlim(right=top_aeroportos.max() * 1.12)
ax.set_xlabel('Número de voos')
ax.set_ylabel('Aeroporto')
ax.set_title('Top 15 aeroportos por movimentação')
fig.tight_layout()
plt.show()

# 2. MAPA 1: GEOPANDAS (Rede de Voos)
df_mapa = df_vra_join.dropna(subset=['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino', 'Latitude Origem', 'Longitude Origem', 'Latitude Destino', 'Longitude Destino']).copy()
arestas_mapa = df_mapa.groupby(['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino']).size().reset_index(name='peso')

G_vis = nx.from_pandas_edgelist(arestas_mapa, source='ICAO Aeródromo Origem', target='ICAO Aeródromo Destino', edge_attr='peso', create_using=nx.DiGraph())

posicoes = {}
for _, row in df_mapa.drop_duplicates('ICAO Aeródromo Origem').iterrows():
    posicoes[row['ICAO Aeródromo Origem']] = (row['Longitude Origem'], row['Latitude Origem'])
for _, row in df_mapa.drop_duplicates('ICAO Aeródromo Destino').iterrows():
    posicoes[row['ICAO Aeródromo Destino']] = (row['Longitude Destino'], row['Latitude Destino'])

brasil = gpd.read_file('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson')
ax = brasil.plot(figsize=(14, 12), edgecolor='black', facecolor='none')
nx.draw_networkx_edges(G_vis, posicoes, ax=ax, alpha=0.15, arrows=False, width=0.5)
nx.draw_networkx_nodes(G_vis, posicoes, ax=ax, node_size=15)
ax.set_title('Rede de voos domésticos — Julho/2026')
ax.set_axis_off()
plt.show()


# 3. MAPA 2: PLOTLY EXPRESS (Interativo)
origem = df_vra_join.groupby(['ICAO Aeródromo Origem', 'Nome Origem', 'UF Origem', 'Latitude Origem', 'Longitude Origem']).size().reset_index(name='Voos').rename(columns={'ICAO Aeródromo Origem': 'ICAO', 'Nome Origem': 'Aeroporto', 'UF Origem': 'UF', 'Latitude Origem': 'Latitude', 'Longitude Origem': 'Longitude'})
destino = df_vra_join.groupby(['ICAO Aeródromo Destino', 'Nome Destino', 'UF Destino', 'Latitude Destino', 'Longitude Destino']).size().reset_index(name='Voos').rename(columns={'ICAO Aeródromo Destino': 'ICAO', 'Nome Destino': 'Aeroporto', 'UF Destino': 'UF', 'Latitude Destino': 'Latitude', 'Longitude Destino': 'Longitude'})

aero_mov = pd.concat([origem, destino]).groupby(['ICAO', 'Aeroporto', 'UF', 'Latitude', 'Longitude'], as_index=False)['Voos'].sum()

fig = px.scatter_map(
    aero_mov, lat='Latitude', lon='Longitude', size='Voos', color='Voos',
    hover_name='Aeroporto', hover_data={'ICAO': True, 'UF': True, 'Voos': True, 'Latitude': False, 'Longitude': False},
    zoom=3.3, center={'lat': -14.2, 'lon': -51.9}, size_max=35, map_style='open-street-map',
    title='Movimentação dos aeroportos — Julho/2026'
)
fig.show()
