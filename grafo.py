import json
import urllib.request

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px

# 1. LEITURA DOS DADOS (Arquivos no mesmo diretório)
df_vra = pd.read_csv('VRA_20267.csv', sep=';', skiprows=1, low_memory=False) #le os dados, separando por ponto e vírgula, pulando a primeira linha e evitando problemas de memória
df_aerodromos = pd.read_csv('AerodromosPublicos.csv', sep=';', skiprows=1, low_memory=False, encoding='ISO-8859-1') #le os dados, separando por ponto e vírgula, pulando a primeira linha e evitando problemas de memória, com codificação específica


# 2. LIMPEZA VRA (voos domésticos regulares)
prefixos_icao_brasil = ('SB', 'SD', 'SI', 'SJ', 'SN', 'SS', 'SW')
df_vra_filtrado = df_vra.drop(columns=['Código Justificativa']).loc[
    lambda df: (
        (df['Situação Voo'] != 'CANCELADO')
        & df['Código Autorização (DI)'].isin(['0', '4', 'C'])
        & df['Código Tipo Linha'].isin(['N', 'C'])
        & df['ICAO Aeródromo Origem'].str.startswith(prefixos_icao_brasil, na=False)
        & df['ICAO Aeródromo Destino'].str.startswith(prefixos_icao_brasil, na=False)
    )
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
print(f"Grau Médio Total (entrada + saída): {(sum(graus) / len(graus)):.2f} conexões por aeroporto")

G_undirected = G.to_undirected()
print(f"Grau Médio na Projeção Não Direcionada: {(2 * G_undirected.number_of_edges() / G_undirected.number_of_nodes()):.2f} conexões por aeroporto")
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

cadastro_aeroportos = df_aerodromos[['Código OACI', 'CIAD', 'Nome', 'Município']].drop_duplicates('Código OACI')
rotulos_aeroportos = {}
for _, aeroporto in cadastro_aeroportos.iterrows():
    nome = aeroporto['Nome']
    if pd.isna(nome) or not str(nome).strip():
        nome = aeroporto['Município']
    if pd.isna(nome) or not str(nome).strip():
        nome = aeroporto['Código OACI']
    cidade = aeroporto['Município']
    if pd.isna(cidade) or not str(cidade).strip():
        cidade = 'cidade não informada'
    uf = str(aeroporto['CIAD'])[:2] if pd.notna(aeroporto['CIAD']) else 'UF não informada'
    rotulos_aeroportos[aeroporto['Código OACI']] = (
        f'{str(nome).strip().title()} ({str(cidade).strip().title()} - {uf})'
    )

# Calcula as métricas usando o grafo direcionado G original
centralidade_entrada = nx.in_degree_centrality(G)
centralidade_saida = nx.out_degree_centrality(G)
exibir_top5("Grau de Entrada (Voos que chegam)", centralidade_entrada)
exibir_top5("Grau de Saída (Voos que partem)", centralidade_saida)
exibir_top5("Closeness Centrality (Proximidade / Fácil acesso)", nx.closeness_centrality(G))
exibir_top5("Betweenness Centrality (Intermediação / Conexões)", nx.betweenness_centrality(G))
exibir_top5("PageRank (Importância Global)", nx.pagerank(G))

# Eigenvector costuma dar erro de convergência em grafos direcionados esparsos,
# usar a versão não-direcionada resolve e mantém a lógica matemática.
exibir_top5("Eigenvector Centrality (Influência da Vizinhança)", nx.eigenvector_centrality(G_undirected, max_iter=1000))

top_entrada = [aeroporto for aeroporto, _ in sorted(centralidade_entrada.items(), key=lambda item: item[1], reverse=True)[:5]]
top_saida = [aeroporto for aeroporto, _ in sorted(centralidade_saida.items(), key=lambda item: item[1], reverse=True)[:5]]
top_centralidade = sorted(
    set(top_entrada + top_saida),
    key=lambda aeroporto: (centralidade_entrada[aeroporto] + centralidade_saida[aeroporto]) / 2,
    reverse=True
)
rotulos = [rotulos_aeroportos.get(aeroporto, aeroporto) for aeroporto in top_centralidade]
valores_entrada = [centralidade_entrada[aeroporto] for aeroporto in top_centralidade]
valores_saida = [centralidade_saida[aeroporto] for aeroporto in top_centralidade]
posicoes = range(len(top_centralidade))
altura_barra = 0.36

fig, ax = plt.subplots(figsize=(14, 8))
barra_entrada = ax.barh(
    [posicao - altura_barra / 2 for posicao in posicoes],
    valores_entrada, height=altura_barra, color='#2563eb', label='Voos que chegam'
)
barra_saida = ax.barh(
    [posicao + altura_barra / 2 for posicao in posicoes],
    valores_saida, height=altura_barra, color='#f97316', label='Voos que partem'
)
ax.bar_label(barra_entrada, fmt='%.4f', padding=3, fontsize=8)
ax.bar_label(barra_saida, fmt='%.4f', padding=3, fontsize=8)
ax.set_yticks(list(posicoes), labels=rotulos)
ax.set_xlabel('Centralidade de grau')
ax.set_title('Top 5 aeroportos por centralidade: grau de entrada e saída')
ax.legend()
ax.grid(axis='x', alpha=0.25)
ax.invert_yaxis()
fig.tight_layout()
plt.show()

# Redes clássicas para comparação: todas usam a projeção não direcionada da rede real.
seed = 42
nos = G_undirected.number_of_nodes()
arestas = G_undirected.number_of_edges()
G_random = nx.gnm_random_graph(n=nos, m=arestas, seed=seed)

k_medio = max(2, round((2 * arestas) / nos))
if k_medio % 2 != 0:
    k_medio -= 1
G_small_world = nx.watts_strogatz_graph(n=nos, k=k_medio, p=0.1, seed=seed)

m_ba = max(1, min(round(arestas / nos), nos - 1))
G_scale_free = nx.barabasi_albert_graph(n=nos, m=m_ba, seed=seed)

def exibir_metricas_modelo(nome, grafo):
    componente = max(nx.connected_components(grafo), key=len)
    grafo_core = grafo.subgraph(componente)
    metricas = {
        'Densidade': nx.density(grafo),
        'Clustering': nx.average_clustering(grafo),
        'Diâmetro': nx.diameter(grafo_core),
        'Distância média': nx.average_shortest_path_length(grafo_core),
    }
    print(f"\n--- {nome} ---")
    print(f"Densidade: {metricas['Densidade']:.4f}")
    print(f"Clustering: {metricas['Clustering']:.4f}")
    print(f"Diâmetro: {metricas['Diâmetro']}")
    print(f"Distância média: {metricas['Distância média']:.4f} saltos")
    return metricas

modelos = [
    ('Rede real', G_undirected),
    ('Erdos-Renyi', G_random),
    ('Watts-Strogatz', G_small_world),
    ('Barabasi-Albert', G_scale_free),
]
cores_modelos = ['#2563eb', '#f97316', '#16a34a', '#dc2626']
resultados_modelos = [
    exibir_metricas_modelo(nome, grafo)
    for nome, grafo in modelos
]

fig, eixos = plt.subplots(2, 2, figsize=(16, 10))
for eixo, metrica in zip(eixos.flat, ['Densidade', 'Clustering', 'Diâmetro', 'Distância média']):
    valores = [resultado[metrica] for resultado in resultados_modelos]
    barras = eixo.bar(
        [nome for nome, _ in modelos], valores,
        color=cores_modelos, width=0.68
    )
    formato = '%.0f' if metrica == 'Diâmetro' else '%.3f'
    eixo.bar_label(barras, fmt=formato, padding=3, fontsize=8)
    eixo.set_title(metrica)
    eixo.tick_params(axis='x', labelsize=8, rotation=25)
    eixo.grid(axis='y', alpha=0.25)
fig.suptitle('Comparação das métricas entre as redes', fontsize=14)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, hspace=0.55, wspace=0.25)
plt.show()

fig, eixos = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
for eixo, (nome, grafo), cor in zip(eixos.flat, modelos, cores_modelos):
    contagem = pd.Series(dict(grafo.degree())).value_counts().sort_index()
    eixo.bar(contagem.index, contagem.values, color=cor, width=0.8)
    eixo.set_title(nome)
    eixo.set_xlabel('Grau')
    eixo.set_ylabel('Número de nós')
    eixo.locator_params(axis='x', nbins=8)
    eixo.grid(axis='y', alpha=0.25)
fig.suptitle('Distribuição de graus', fontsize=14)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, hspace=0.45, wspace=0.25)
plt.show()



# Mapas

# Junções de Tabelas para Criação dos Mapas
aero_cols = ['Código OACI', 'CIAD', 'Nome', 'Município', 'UF', 'LATGEOPOINT', 'LONGEOPOINT']
# O cadastro completo é usado para identificar nomes e coordenadas dos voos.
df_aero = df_aerodromos[aero_cols].copy()

# Rename Origem
df_aero_origem = df_aero.rename(columns={
    'Código OACI': 'Código OACI Origem', 'CIAD': 'CIAD Origem', 'Nome': 'Nome Origem',
    'Município': 'Município Origem', 'UF': 'UF Origem', 
    'LATGEOPOINT': 'Latitude Origem', 'LONGEOPOINT': 'Longitude Origem'
})

# Merge Origem
df_vra_join = df_voos.merge(
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


# 1. GRÁFICO DE BARRAS: Top 15 Aeroportos em Quantidade de Voos (para comparação com aeroportos mais centrais)
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

"""
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
"""