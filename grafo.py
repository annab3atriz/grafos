import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import urllib.request
import json

# 1. LEITURA DOS DADOS (Arquivos no mesmo diretório)
df_vra = pd.read_csv('VRA_20267.csv', sep=';', skiprows=1, low_memory=False) #le os dados, separando por ponto e vírgula, pulando a primeira linha e evitando problemas de memória
df_aerodromos = pd.read_csv('AerodromosPublicos.csv', sep=';', skiprows=1, low_memory=False, encoding='ISO-8859-1') #le os dados, separando por ponto e vírgula, pulando a primeira linha e evitando problemas de memória, com codificação específica


# 2. LIMPEZA VRA (Voos)
df_vra_filtrado = (
    df_vra
    .drop(columns=['Código Justificativa']) #remove coluna desnecessária
    .loc[ #filtra linhas com base em condições específicas
        lambda df: ( #só mantem o registro se todas as condições forem verdadeiras
            (df['Situação Voo'] != 'CANCELADO') & #mantém apenas voos não cancelados
            (df['Código Autorização (DI)'].isin(['0', '4', 'C'])) & #mantém apenas voos com autorização específica (voos regulares, voos de carga e voos de passageiros)
            (df['Código Tipo Linha'].isin(['N', 'C'])) #mantém apenas voos de tipo específico (voos comerciais com passageiros e voos de carga)
        )
    ]
)

# 3. LIMPEZA AERÓDROMOS
df_aerodromos_filtrado = df_aerodromos.drop(columns=['Portaria de Registro', 'Link Portaria']) #remove colunas desnecessárias
df_aerodromos_filtrado = df_aerodromos_filtrado[df_aerodromos_filtrado['Situação'] != 'Interditado'] #mantém apenas aeroportos não interditados
df_aerodromos_filtrado['Validade do Registro'] = pd.to_datetime(df_aerodromos_filtrado['Validade do Registro'], format='%d/%m/%Y', errors='coerce') #converte a coluna para datetime

comparison_date = pd.to_datetime('01/07/2026', format='%d/%m/%Y') #define a data de comparação (01/07/2026) para filtrar registros válidos
df_aerodromos_filtrado = df_aerodromos_filtrado[
    df_aerodromos_filtrado['Validade do Registro'].isna() | #mantém registros com data de validade nula
    (df_aerodromos_filtrado['Validade do Registro'] > comparison_date) #se a data de validade for nula ou maior que 01/07/2026, mantém o registro
]

# 1. CONSTRUÇÃO DA REDE PURA (Sem coordenadas)
df_voos = df_vra_filtrado.dropna(subset=['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino']).copy() #remove linhas com valores nulos nas colunas de origem e destino
df_voos = df_voos[df_voos['ICAO Aeródromo Origem'] != df_voos['ICAO Aeródromo Destino']] #remove voos que partem e chegam no mesmo aeroporto

# Cria os pesos (número de voos)
arestas_calculo = df_voos.groupby(['ICAO Aeródromo Origem', 'ICAO Aeródromo Destino']).size().reset_index(name='peso') #registra o número de voos entre cada par de aeroportos (origem e destino) e cria uma coluna 'peso' com esse valor

# Representação computacional
G = nx.from_pandas_edgelist(
    arestas_calculo,
    source='ICAO Aeródromo Origem',
    target='ICAO Aeródromo Destino',
    edge_attr='peso',
    create_using=nx.DiGraph()
)

# 2. CARACTERIZAÇÃO ESTRUTURAL

print(f"Número de Nós (Aeroportos): {G.number_of_nodes()}")
print(f"Número de Arestas (Rotas): {G.number_of_edges()}")
print(f"Densidade da Rede: {nx.density(G):.4f}")

graus = [grau for no, grau in G.degree()]
print(f"Grau Médio da Rede: {(sum(graus) / len(graus)):.2f} conexões por aeroporto")

G_undirected = G.to_undirected()
print(f"Coeficiente de Aglomeração (Clustering): {nx.average_clustering(G_undirected):.4f}")

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
    top5 = sorted(dict_centralidade.items(), key=lambda x: x[1], reverse=True)[:5] #pega os 5 primeiros elementos da lista ordenada
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

# Redes Clássicas para Comparação
n = G.number_of_nodes();
m = G.number_of_edges();
G_random = nx.gnm_random_graph(n=n, m=m, directed=True)
G_small_world = nx.watts_strogatz_graph(n=n, k=13, p=0.1)
G_scale_free = nx.barabasi_albert_graph(n=n, m=m)

print(f"--- Random ---")
print(f"Densidade: {nx.density(G_random):.4f}")
print(f"Coeficiente de Aglomeração (Clustering): {nx.average_clustering(G_random):.4f}")
    
# Isola o maior componente conectado para evitar erros de nós isolados no cálculo de distância
maior_componente = max(nx.connected_components(G_random), key=len)
grafo_core = G_random.subgraph(maior_componente)
    
print(f"Diâmetro: {nx.diameter(grafo_core)}")
print(f"Distância Média: {nx.average_shortest_path_length(grafo_core):.4f} saltos\n")
grafo_core = G_random.subgraph(maior_componente)
    
print(f"Diâmetro: {nx.diameter(grafo_core)}")
print(f"Distância Média: {nx.average_shortest_path_length(grafo_core):.4f} saltos\n")



# ------------------------
# -------- Mapas ---------
# ------------------------

# Junções de Tabelas para Criação dos Mapas
aero_cols = ['Código OACI', 'CIAD', 'Nome', 'Município', 'UF', 'LATGEOPOINT', 'LONGEOPOINT']
df_aero = df_aerodromos_filtrado[aero_cols].copy()

# Rename Origem
df_aero_origem = df_aero.rename(columns={
    'Código OACI': 'Código OACI Origem', 'CIAD': 'CIAD Origem', 'Nome': 'Nome Origem',
    'Município': 'Município Origem', 'UF': 'UF Origem', 
    'LATGEOPOINT': 'Latitude Origem', 'LONGEOPOINT': 'Longitude Origem'
})

# Merge Origem
df_vra_join = df_vra_filtrado.merge(
    df_aero_origem, left_on='ICAO Aeródromo Origem', right_on='Código OACI Origem', how='left', validate='many_to_one'
)

# Rename Destino
df_aero_destino = df_aero.rename(columns={
    'Código OACI': 'Código OACI Destino', 'CIAD': 'CIAD Destino', 'Nome': 'Nome Destino',
    'Município': 'Município Destino', 'UF': 'UF Destino', 
    'LATGEOPOINT': 'Latitude Destino', 'LONGEOPOINT': 'Longitude Destino'
})

# Merge Destino
df_vra_join = df_vra_join.merge(
    df_aero_destino, left_on='ICAO Aeródromo Destino', right_on='Código OACI Destino', how='left', validate='many_to_one'
)

df_vra_join['UF Origem'] = df_vra_join['CIAD Origem'].str[:2]
df_vra_join['UF Destino'] = df_vra_join['CIAD Destino'].str[:2]

print("Base pronta para gerar os mapas!")


# 1. GRÁFICO DE BARRAS: Top 15 Aeroportos em Quantidade de Voos (para comparação com aeroportos mais centrais)
top_aeroportos = pd.concat([df_vra_join['ICAO Aeródromo Origem'], df_vra_join['ICAO Aeródromo Destino']]).value_counts().head(15).sort_values()
top_aeroportos.plot(kind='barh', figsize=(10, 6))
plt.xlabel('Número de voos')
plt.ylabel('Aeroporto (OACI)')
plt.title('Top 15 aeroportos por movimentação')
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