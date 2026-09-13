# Analise de uma rede de aeroportos

Projeto da disciplina de Grafos para caracterizar uma rede real de voos
domesticos brasileiros e comparar sua estrutura com modelos classicos de
redes complexas.

## Pergunta de investigacao

Quais propriedades estruturais tornam a rede de voos uma rede complexa e ate
que ponto essas propriedades podem ser reproduzidas pelos modelos
Erdos-Renyi, Watts-Strogatz e Barabasi-Albert?

## Dados

O projeto espera os seguintes arquivos na raiz:

- `VRA_20267.csv`: registros de voos da ANAC.
- `AerodromosPublicos.csv`: cadastro de aerodromos publicos, usado para nomes,
  municipios, estados e coordenadas.

Sao considerados somente voos:

- domesticos, com `Codigo Tipo Linha` igual a `N` ou `C`;
- regulares, com `Codigo Autorizacao (DI)` igual a `0`, `4` ou `C`;
- nao cancelados;
- com origem e destino identificados por prefixos OACI de aerodromos
  brasileiros.

Cada no representa um aeroporto. Cada aresta direcionada representa uma rota,
e seu peso e o numero de voos registrados entre origem e destino.

## Instalacao

Recomenda-se usar um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente e instale as dependencias:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execucao

O fluxo principal carrega os dados relativos ao diretorio do script, calcula
as metricas, exibe os principais aeroportos por centralidade e compara a rede
real com tres modelos classicos:

```bash
python analise_rede.py
```

Tambem e possivel executar `grafo.py`, embora esse arquivo contenha uma
implementacao mais extensa, com analises e visualizacoes adicionais:

```bash
python grafo.py
```

O notebook `TrabGrafos1.ipynb` pode ser aberto no Jupyter ou no VS Code para
executar a analise interativamente.

## Analises realizadas

- numero de nos e arestas;
- densidade e grau medio;
- clustering, distancia media e diametro da maior componente conexa;
- centralidades de entrada, saida, closeness, betweenness, eigenvector e
  PageRank;
- distribuicao de graus;
- comparacao com Erdos-Renyi, Watts-Strogatz e Barabasi-Albert.

As redes de comparacao usam a projecao nao direcionada da rede real e a
semente aleatoria `42`, permitindo reproduzir os resultados.

## Estrutura

```text
.
|-- AerodromosPublicos.csv
|-- VRA_20267.csv
|-- analise_rede.py       # fluxo principal modularizado
|-- dados_voo.py          # funcoes auxiliares de dados
|-- grafo.py              # analise completa e visualizacoes
|-- TrabGrafos1.ipynb     # execucao interativa
|-- Description.md        # descricao do problema e dos criterios
`-- requirements.txt
```