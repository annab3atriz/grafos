Projeto 1 — Caracterização e Modelagem de uma Rede Complexa Real

Objetivo
Investigar as propriedades estruturais de uma rede complexa real e verificar em que medida sua estrutura pode ser explicada por modelos clássicos de redes.
Problema
Cada grupo deverá escolher uma rede real e formular uma questão de investigação relacionada à sua estrutura. Possíveis redes:

rede de aeroportos; 

Critérios de filtragem dos voos

Foram considerados somente voos domésticos, regulares e não cancelados, conforme a descrição de variáveis da ANAC:

- `Código Tipo Linha` igual a `N` (doméstica mista) ou `C` (doméstica cargueira);
- `Código Autorização (DI)` igual a `0`, `4` ou `C` (categoria regular);
- `Situação Voo` diferente de `CANCELADO`;
- origem e destino com prefixos OACI de aeródromos brasileiros.

O código `C` aparece nas duas colunas porque representa uma linha doméstica cargueira em `Código Tipo Linha` e uma autorização regular em `Código Autorização (DI)`. Voos internacionais (`I` e `G`) e categorias não regulares não fazem parte da rede analisada.

Os pesos das arestas correspondem ao número de voos em cada rota. As métricas topológicas e de centralidade usam a existência das conexões; o peso é mantido para representar o volume da rota e pode ser usado em análises ponderadas específicas.

Atividades
1. Construção da rede
Os alunos deverão:
definir os nós e as arestas; 
definir se a rede é direcionada/não direcionada; 
definir se é ponderada/não ponderada; 
obter e documentar os dados; 
construir a representação computacional da rede. 

2. Caracterização estrutural
Calcular e interpretar:
número de nós e arestas; 
grau médio; 
distribuição de graus; 
densidade; 
clustering coefficient; 
average path length; 
diâmetro; 
componentes conexos; 

3. Centralidade
Investigar os nós mais importantes usando:
degree centrality; 
closeness; 
betweenness; 
eigenvector centrality; 
PageRank, quando aplicável. 
Uma questão interessante é verificar se os diferentes conceitos de centralidade identificam os mesmos "nós importantes".

4. Comparação com modelos
Gerar redes equivalentes usando, por exemplo:
Erdős–Rényi; 
Watts–Strogatz; 
Barabási–Albert. 
Comparar:
distribuição de grau; 
clustering; 
path length; 

Entregáveis
código; 
visualização da rede; 

Pergunta central
Quais propriedades estruturais tornam a rede estudada uma rede complexa, e até que ponto essas propriedades podem ser reproduzidas por modelos clássicos?
aleatoria pequeno mundo livre escala

Comparação com modelos clássicos

Para comparar a estrutura, são gerados modelos Erdős-Rényi, Watts-Strogatz e Barabási-Albert com o mesmo número de nós e aproximadamente o mesmo número de arestas da projeção não direcionada da rede real. A comparação considera densidade, clustering, diâmetro, distância média e distribuição de graus. A semente aleatória `42` torna os resultados reproduzíveis.

