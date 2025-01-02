# Graph Neural Alchemist: Um Framework Modular para Classificação de Séries Temporais usando Redes Neurais em Grafos

## Preâmbulo

O Graph Neural Alchemist é um framework modular para classificação de séries temporais representadas como grafos que permite utilizar qualquer abordagem de representação de séries temporais como grafos e treinar com várias arquiteturas de GNN.

Este framework foi utilizado em parte para o artigo ["Graph Neural Alchemist: An innovative fully modular architecture for time series-to-graph classification"](https://doi.org/10.48550/arXiv.2410.09307).

O framework permite acompanhar o treinamento através do PyTorch Lightning, que oferece poder para visualizar as métricas de cada época, bem como flags específicas para debugar o modelo. Para visualizar em tempo real o desempenho dos modelos a cada experimento, é possível acompanhar pelo TensorBoard, uma vez que esse framework implementa também essa ferramenta.

## Abordagens Implementadas

1. **Grafos de Visibilidade (VG)**
   - Construção de grafos baseada na visibilidade entre pontos da série temporal
   - Cada ponto da série é um nó, conectado se houver "visibilidade" entre os pontos
   - Features dos nós incluem PageRank, grau do nó e valor do sinal
   - Suporta grafos direcionados e não-direcionados
   - Referência: [From time series to complex networks: The visibility graph](https://doi.org/10.1073/pnas.0709247105)

2. **Grafos de Padrões Ordinais (GTPO)**
   - Converte séries temporais em grafos baseados em padrões ordinais
   - Cada nó representa um padrão ordinal específico
   - Arestas representam transições entre padrões consecutivos
   - Features dos nós incluem PageRank e grau do nó
   - Referência: [Ordinal patterns for time series analysis](https://doi.org/10.1038/s41598-017-08245-x)

3. **SimTSC (Similarity-based Time Series Classification)**
   - Classificação baseada em similaridade entre séries temporais
   - Constrói um grafo onde cada nó é uma série temporal
   - Arestas são criadas entre séries similares usando DTW
   - Usa GNN para classificação de nós
   - Referência: [SimTSC: A Novel Geometric Deep Learning Framework for Time Series Classification](https://doi.org/10.48550/arXiv.2201.01413)

4. **Time2Graph/Time2Graph+**
   - Extrai shapelets cônscios ao tempo das séries
   - Constrói grafos onde nós são shapelets extraídos
   - Features dos nós são aprendidas durante o treinamento
   - Referência: [Time2Graph: Revisiting Time Series Modeling with Dynamic Shapelets](https://doi.org/10.1609/aaai.v34i04.5769)

5. **GTPO Codificado (Encoded Graph Time Pattern Ordinal)**
   - Extensão do GTPO com codificação one-hot dos padrões
   - Representa padrões ordinais como vetores binários
   - Mantém a estrutura de transições do GTPO
   - Permite representação mais rica dos padrões

6. **Grafos de Covariância**
   - Constrói grafos baseados na correlação de Pearson entre séries
   - Nós representam séries temporais individuais
   - Arestas são criadas quando a correlação excede um limiar R
   - Útil para análise de séries temporais multivariadas

## Arquiteturas de GNN Disponíveis

1. **SAGE_MLPP_4layer**
   - 4 camadas GraphSAGE seguidas de MLP
   - Readout por média dos nós
   - Suporta diferentes tipos de agregação
   - Ideal para classificação de grafos inteiros

2. **GAT_MLPP**
   - Múltiplas camadas de Graph Attention Network (GAT)
   - Número configurável de camadas e cabeças de atenção
   - MLP posterior para classificação
   - Readout por média dos nós
   - Captura importância relativa entre nós vizinhos

3. **SAGE_NodeClassification**
   - 3 camadas GraphSAGE para classificação de nós
   - Específico para abordagens como SimTSC
   - Otimizado para predição no nível do nó

4. **SimTSC_GCN**
   - Combina ResNet com Graph Convolutional Network
   - 3 blocos ResNet para processamento de séries temporais
   - 3 camadas GCN para aprendizado de relações entre amostras
   - Dropout configurável entre camadas GCN
   - Baseado no artigo original do SimTSC

5. **SimTSC_SAGE**
   - Variante do SimTSC usando GraphSAGE
   - 3 blocos ResNet + 3 camadas GraphSAGE
   - Suporta diferentes tipos de agregação
   - Mais flexível que a versão GCN para grafos grandes

## Extensibilidade dos Modelos

O framework foi projetado para ser facilmente extensível. Os modelos podem ser modificados ou novos modelos podem ser criados seguindo estas diretrizes:

1. **Modificando Arquiteturas Existentes**
   - Herde de uma classe existente e sobrescreva os métodos relevantes
   - Ajuste número de camadas alterando a construção no `__init__`
   - Modifique estratégias de pooling no método `forward`
   
   Exemplo de personalização:
   ```python
   class CustomSAGE(SAGE_MLPP_4layer):
       def __init__(self, args):
           super().__init__(args)
           # Adicione mais camadas
           self.conv5 = SAGEConv(self.h_feats, self.h_feats, self.agg_type)
           
       def forward(self, graph, node_features):
           # ... processamento das camadas existentes ...
           
           # Use pooling diferente
           readout_h = dgl.max_nodes(graph, 'h')  # max pooling ao invés de mean
           return F.log_softmax(h, dim=1)
   ```

2. **Criando Novos Modelos**
   - Herde de `pl.LightningModule`
   - Implemente no mínimo `__init__` e `forward`
   - Siga o padrão de interface dos modelos existentes
   
   Exemplo básico:
   ```python
   class NewGNNModel(pl.LightningModule):
       def __init__(self, args):
           super().__init__()
           self.in_feats = args.num_features
           self.h_feats = args.nhid
           self.num_classes = args.num_classes
           # Defina suas camadas aqui
           
       def forward(self, graph, node_features):
           # Implemente a lógica do modelo
           return output
   ```

3. **Modificando Estratégias de Agregação**
   - Para modelos SAGE: modifique o parâmetro `agg_type` 
   - Para GAT: ajuste `num_heads` e parâmetros de atenção
   - Para GCN: modifique a função de normalização da matriz de adjacência

4. **Arquiteturas Híbridas**
   - Combine diferentes tipos de camadas (como em SimTSC)
   - Misture estratégias de pooling
   - Adicione camadas residuais ou skip connections

5. **Dicas de Implementação**
   - Mantenha compatibilidade com a interface existente
   - Documente parâmetros e comportamentos específicos
   - Use os decoradores do PyTorch Lightning para hooks de treinamento
   - Implemente métodos de configuração de otimizador se necessário

Todos os modelos devem ser compatíveis com o pipeline de treinamento do PyTorch Lightning e seguir as convenções de entrada/saída do framework.

## Requisitos e Instalação

### Dependências Principais
- Python 3.8+
- PyTorch >= 1.9.0
- DGL (Deep Graph Library) >= 0.9.0
- PyTorch Lightning >= 2.0.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/graph-neural-alchemist.git
cd graph-neural-alchemist

# Instale as dependências
pip install -r requirements.txt
```

## Estrutura dos Dados

### Formato dos Arquivos
- Arquivos TSV (Tab-Separated Values)
- Primeira coluna: rótulo da classe (inteiro)
- Colunas seguintes: valores da série temporal (float)
- Nomenclatura: `DATASET_TRAIN.tsv` e `DATASET_TEST.tsv`

Exemplo:
```
1    -0.752  -0.897  -0.889  ...
0     0.321   0.453   0.472  ...
```

### Organização dos Datasets
```
data/
├── dataset1/
│   ├── dataset1_TRAIN.tsv
│   └── dataset1_TEST.tsv
├── dataset2/
│   ├── dataset2_TRAIN.tsv
│   └── dataset2_TEST.tsv
```

## Execução de Experimentos

### 1. Configuração
Defina os datasets a serem processados em `parameters.json`:
```json
{
  "datasets": ["dataset1", "dataset2"]  
}
```

### 2. Execução Básica
```bash
python run.py --strategy vg --model SAGE_MLPP_4layer --batch_size 32  --nhid 128 --epochs 250 --save_dir vg_sage_mlpp_4layer --root_path data/dataset1
```

### 3. Parâmetros Principais
- `--strategy`: Método de construção do grafo
  - `vg`: Grafos de Visibilidade
  - `op`: Padrões Ordinais
  - `simtsc`: SimTSC
  - `time2graph`: Time2Graph/Time2Graph+
  - `encoded_gtpo`: GTPO Codificado
  - `covariance`: Grafos de Covariância

- `--model`: Arquitetura da GNN
  - `SAGE_MLPP_4layer`
  - `GAT_MLPP`
  - `simTSC_GCN`
  - `simTSC_SAGE`

### 4. Monitoramento
```bash
# Inicie o TensorBoard
tensorboard --logdir lightning_logs

# Acesse via navegador
http://localhost:6006
```

### 5. Debugando

O framework possui um logger que pode ser acessado para debugar o modelo. Para isso, basta acessar o diretório `logs` e abrir o arquivo no formato `YYYY-MM-DD_main.log` (exemplo: `2024-08-28_main.log`). Todas as saídas de debug e eventuais erros são salvas neste arquivo.

Além disso, é possível utilizar funções de debug específicas do PyTorch Lightning para debugar o modelo. Para isso, as seguintes flags podem ser utilizadas ao executar o script:

- `--detect_anomaly`: Ativa o detector de anomalias do PyTorch para identificar operações inválidas ou instabilidades numéricas durante o treinamento. Útil para identificar erros de gradiente.
- `--fast_dev_run_batches N`: Executa apenas N batches de treino e teste para validar rapidamente o pipeline completo. Útil para testar se o pipeline está funcionando corretamente.
- `--overfit_batches N`: Treina repetidamente em N batches para verificar se o modelo tem capacidade de overfit. Útil para verificar se o modelo está generalizando bem.

Para mais detalhes sobre debugging no PyTorch Lightning, consulte: https://lightning.ai/docs/pytorch/stable/debug/debugging.html


Exemplo de uso:
```bash
python run.py --strategy vg --model SAGE_MLPP_4layer --batch_size 32 --nhid 128 --epochs 250 --save_dir vg_sage_mlpp_4layer --root_path data/dataset1 --detect_anomaly --fast_dev_run_batches 10 --overfit_batches 10
```

## Guia de Contribuição

Esse projeto é de código aberto e aceita contribuições. Para isso, siga os seguintes passos:

### 1. Issues e Discussões
- Abra issues para reportar bugs ou sugerir melhorias
- Use labels apropriados: `bug`, `enhancement`, `documentation`
- Forneça exemplos reproduzíveis quando reportar problemas

### 2. Processo de Desenvolvimento
```bash
# Fork e clone
git clone https://github.com/seu-usuario/graph-neural-alchemist.git

# Crie uma branch para sua feature
git checkout -b feature/nova-funcionalidade

# Commit com mensagens descritivas
git commit -m "feat: adiciona nova estratégia de grafo"

# Push e Pull Request
git push origin feature/nova-funcionalidade
```

### 3. Padrões de Código
- Siga PEP 8 para estilo de código Python
- Documente classes e métodos usando docstrings:
```python
class NovaEstrategia:
    """
    Implementa uma nova estratégia de conversão série-grafo.

    Parâmetros
    ----------
    param1 : tipo
        Descrição do parâmetro
    param2 : tipo
        Descrição do parâmetro

    Atributos
    ----------
    attr1 : tipo
        Descrição do atributo
    """
```

### 4. Modularização
- Mantenha separação clara entre:
  - Estratégias de construção e armazenamento de grafos (`datasets/`)
  - Arquiteturas de modelos (`models/`)  
- Use herança e composição apropriadamente
- Implemente interfaces consistentes

### 5. Versionamento
- Use tags semânticas (MAJOR.MINOR.PATCH)
- Mantenha um CHANGELOG.md atualizado
- Documente breaking changes

### 6. Testes
- Adicione testes unitários para novas funcionalidades
- Verifique compatibilidade com diferentes versões de dependências
- Execute testes antes de submeter PRs

### 7. Documentação
- Atualize o README.md quando adicionar funcionalidades
- Mantenha exemplos de uso atualizados
- Documente parâmetros e comportamentos específicos

## Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## Citação

Se você usar este código em sua pesquisa, por favor cite:

```bibtex
@article{graphneuralalchemist2023,
  title={Graph Neural Alchemist: An innovative fully modular architecture for time series-to-graph classification},
  author={Coelho, Paulo H. and outros},
  journal={arXiv preprint arXiv:2410.09307},
  year={2023}
}
```

## Contato

Para questões e sugestões:
- Abra uma issue no repositório
- Email: paulohdscoelho@dcc.ufmg.br
- Linkedin: https://www.linkedin.com/in/paulohdscoelho/
