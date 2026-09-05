import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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

# 3. Desenha o grafo usando as posições reais
plt.figure(figsize=(12, 10))

nx.draw(
    G, 
    pos=posicoes,           # Usa as coordenadas geográficas
    node_size=30,           
    node_color='red',       
    edge_color='gray',      
    with_labels=False,      
    alpha=0.5               
)

plt.title("Rede de Aeroportos no Brasil - Mapa Geográfico")
plt.show()