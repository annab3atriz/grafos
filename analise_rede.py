from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
VRA_FILE = BASE_DIR / "VRA_20267.csv"
AERODROMOS_FILE = BASE_DIR / "AerodromosPublicos.csv"
PREFIXOS_ICAO_BRASIL = ("SB", "SD", "SI", "SJ", "SN", "SS", "SW")
SEED = 42


def carregar_dados():
    voos = pd.read_csv(VRA_FILE, sep=";", skiprows=1, low_memory=False)
    aerodromos = pd.read_csv(
        AERODROMOS_FILE,
        sep=";",
        skiprows=1,
        low_memory=False,
        encoding="ISO-8859-1",
    )
    return voos, aerodromos


def filtrar_voos(voos):
    return voos.loc[
        (voos["Situação Voo"] != "CANCELADO")
        & voos["Código Autorização (DI)"].isin(["0", "4", "C"])
        & voos["Código Tipo Linha"].isin(["N", "C"])
        & voos["ICAO Aeródromo Origem"].str.startswith(PREFIXOS_ICAO_BRASIL, na=False)
        & voos["ICAO Aeródromo Destino"].str.startswith(PREFIXOS_ICAO_BRASIL, na=False)
    ].dropna(subset=["ICAO Aeródromo Origem", "ICAO Aeródromo Destino"])


def construir_grafo(voos):
    voos = voos.loc[
        voos["ICAO Aeródromo Origem"] != voos["ICAO Aeródromo Destino"]
    ]
    arestas = (
        voos.groupby(["ICAO Aeródromo Origem", "ICAO Aeródromo Destino"])
        .size()
        .reset_index(name="peso")
    )
    grafo = nx.from_pandas_edgelist(
        arestas,
        source="ICAO Aeródromo Origem",
        target="ICAO Aeródromo Destino",
        edge_attr="peso",
        create_using=nx.DiGraph,
    )
    return grafo, voos, arestas


def maior_componente(grafo):
    nos = max(nx.connected_components(grafo), key=len)
    return grafo.subgraph(nos)


def imprimir_estrutura(grafo):
    nao_direcionado = grafo.to_undirected()
    componente = maior_componente(nao_direcionado)
    print(f"Nós: {grafo.number_of_nodes()}")
    print(f"Arestas direcionadas: {grafo.number_of_edges()}")
    print(f"Densidade direcionada: {nx.density(grafo):.4f}")
    print(f"Grau médio total: {sum(dict(grafo.degree()).values()) / grafo.number_of_nodes():.2f}")
    print(f"Grau médio não direcionado: {2 * nao_direcionado.number_of_edges() / nao_direcionado.number_of_nodes():.2f}")
    print(f"Clustering: {nx.average_clustering(nao_direcionado):.4f}")
    print(f"Diâmetro: {nx.diameter(componente)}")
    print(f"Distância média: {nx.average_shortest_path_length(componente):.4f}")


def top_centralidade(nome, valores, limite=5):
    print(f"\n{nome}:")
    for aeroporto, valor in sorted(valores.items(), key=lambda item: item[1], reverse=True)[:limite]:
        print(f" - {aeroporto}: {valor:.4f}")


def calcular_centralidades(grafo):
    nao_direcionado = grafo.to_undirected()
    centralidades = {
        "Entrada": nx.in_degree_centrality(grafo),
        "Saída": nx.out_degree_centrality(grafo),
        "Closeness": nx.closeness_centrality(grafo),
        "Betweenness": nx.betweenness_centrality(grafo),
        "PageRank": nx.pagerank(grafo),
        "Eigenvector": nx.eigenvector_centrality(nao_direcionado, max_iter=1000),
    }
    for nome, valores in centralidades.items():
        top_centralidade(nome, valores)
    return centralidades


def rotulos_aeroportos(aerodromos):
    cadastro = aerodromos[["Código OACI", "CIAD", "Nome", "Município"]].drop_duplicates("Código OACI")
    rotulos = {}
    for _, aeroporto in cadastro.iterrows():
        nome = aeroporto["Nome"]
        cidade = aeroporto["Município"]
        if pd.isna(nome) or not str(nome).strip():
            nome = cidade
        if pd.isna(nome) or not str(nome).strip():
            nome = aeroporto["Código OACI"]
        if pd.isna(cidade) or not str(cidade).strip():
            cidade = "cidade não informada"
        uf = str(aeroporto["CIAD"])[:2] if pd.notna(aeroporto["CIAD"]) else "UF não informada"
        rotulos[aeroporto["Código OACI"]] = f"{str(nome).strip().title()} ({str(cidade).strip().title()} - {uf})"
    return rotulos


def plotar_centralidade(centralidades, aerodromos):
    entrada = centralidades["Entrada"]
    saida = centralidades["Saída"]
    selecionados = set(
        list(dict(sorted(entrada.items(), key=lambda item: item[1], reverse=True)[:5]))
        + list(dict(sorted(saida.items(), key=lambda item: item[1], reverse=True)[:5]))
    )
    aeroportos = sorted(selecionados, key=lambda aeroporto: (entrada[aeroporto] + saida[aeroporto]) / 2, reverse=True)
    rotulos = rotulos_aeroportos(aerodromos)
    posicoes = range(len(aeroportos))
    altura = 0.36

    fig, ax = plt.subplots(figsize=(14, 8))
    chegada = ax.barh(
        [posicao - altura / 2 for posicao in posicoes],
        [entrada[aeroporto] for aeroporto in aeroportos],
        height=altura, color="#2563eb", label="Voos que chegam",
    )
    partida = ax.barh(
        [posicao + altura / 2 for posicao in posicoes],
        [saida[aeroporto] for aeroporto in aeroportos],
        height=altura, color="#f97316", label="Voos que partem",
    )
    ax.bar_label(chegada, fmt="%.4f", padding=3, fontsize=8)
    ax.bar_label(partida, fmt="%.4f", padding=3, fontsize=8)
    ax.set_yticks(list(posicoes), labels=[rotulos.get(aeroporto, aeroporto) for aeroporto in aeroportos])
    ax.set_xlabel("Centralidade de grau")
    ax.set_title("Top aeroportos por centralidade: grau de entrada e saída")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


def criar_modelos(grafo):
    base = grafo.to_undirected()
    nos = base.number_of_nodes()
    arestas = base.number_of_edges()
    k = max(2, round(2 * arestas / nos))
    k -= k % 2
    return [
        ("Rede real", base),
        ("Erdos-Renyi", nx.gnm_random_graph(nos, arestas, seed=SEED)),
        ("Watts-Strogatz", nx.watts_strogatz_graph(nos, k, 0.1, seed=SEED)),
        ("Barabasi-Albert", nx.barabasi_albert_graph(nos, max(1, round(arestas / nos)), seed=SEED)),
    ]


def metricas_modelo(grafo):
    componente = maior_componente(grafo)
    return {
        "Densidade": nx.density(grafo),
        "Clustering": nx.average_clustering(grafo),
        "Diâmetro": nx.diameter(componente),
        "Distância média": nx.average_shortest_path_length(componente),
    }


def comparar_modelos(modelos):
    resultados = []
    for nome, grafo in modelos:
        metricas = metricas_modelo(grafo)
        resultados.append((nome, grafo, metricas))
        print(f"\n--- {nome} ---")
        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

    fig, eixos = plt.subplots(2, 2, figsize=(16, 10))
    for eixo, metrica in zip(eixos.flat, ["Densidade", "Clustering", "Diâmetro", "Distância média"]):
        nomes = [nome for nome, _, _ in resultados]
        valores = [metricas[metrica] for _, _, metricas in resultados]
        barras = eixo.bar(nomes, valores, color=["#2563eb", "#f97316", "#16a34a", "#dc2626"])
        eixo.bar_label(barras, fmt="%.3f", fontsize=8, padding=3)
        eixo.set_title(metrica)
        eixo.tick_params(axis="x", labelsize=8, rotation=25)
        eixo.grid(axis="y", alpha=0.25)
    fig.suptitle("Comparação das métricas entre as redes")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, hspace=0.55, wspace=0.25)
    plt.show()

    fig, eixos = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for eixo, (nome, grafo, _) in zip(eixos.flat, resultados):
        graus = pd.Series(dict(grafo.degree())).value_counts().sort_index()
        eixo.bar(graus.index, graus.values, width=0.8)
        eixo.set_title(nome)
        eixo.set_xlabel("Grau")
        eixo.set_ylabel("Número de nós")
        eixo.locator_params(axis="x", nbins=8)
        eixo.grid(axis="y", alpha=0.25)
    fig.suptitle("Distribuição de graus")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, hspace=0.45, wspace=0.25)
    plt.show()


def main():
    dados_voos, aerodromos = carregar_dados()
    voos_filtrados = filtrar_voos(dados_voos)
    grafo, voos_rede, arestas = construir_grafo(voos_filtrados)
    print(f"Voos filtrados: {len(voos_filtrados)}")
    print(f"Voos usados na rede: {len(voos_rede)}")
    imprimir_estrutura(grafo)
    print("\n--- Centralidades ---")
    centralidades = calcular_centralidades(grafo)
    plotar_centralidade(centralidades, aerodromos)
    comparar_modelos(criar_modelos(grafo))


if __name__ == "__main__":
    main()
