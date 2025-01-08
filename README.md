# Graph Neural Alchemist: Um Framework Modular para Classificação de Séries Temporais usando Redes Neurais em Grafos

## Preâmbulo

O Graph Neural Alchemist é um framework modular para classificação de séries temporais representadas como grafos que permite utilizar qualquer abordagem de representação de séries temporais como grafos e treinar com várias arquiteturas de GNN.

Este framework foi utilizado em parte para o artigo ["Graph Neural Alchemist: An innovative fully modular architecture for time series-to-graph classification"](https://doi.org/10.48550/arXiv.2410.09307).

O framework permite acompanhar o treinamento através do PyTorch Lightning, que oferece poder para visualizar as métricas de cada época, bem como flags específicas para debugar o modelo. Para visualizar em tempo real o desempenho dos modelos a cada experimento, bem como as métricas de avaliação f1-score, matrizes de confusão, entre outras é possível acompanhar pelo TensorBoard.

## Sumário

- [Abordagens Implementadas](#abordagens-implementadas)  
- [Arquiteturas de GNN Disponíveis](#arquiteturas-de-gnn-disponíveis) 
- [Extensibilidade dos Modelos](#extensibilidade-dos-modelos)
- [Requisitos e Instalação](#requisitos-e-instalação)
- [Estrutura dos Dados](#estrutura-dos-dados)
- [Execução de Experimentos](#execução-de-experimentos)
- [Guia de Contribuição](#guia-de-contribuição)
- [Licença](#licença)
- [Citação](#citação)
- [Contato](#contato)

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

### Implementando Novas Estratégias de Transformação

O framework permite implementar novas estratégias de transformação de séries temporais em grafos através da extensão da classe `dgl.data.DGLDataset`.

É fundamental que essa herança seja feita para que o pipeline funcione corretamente, pois os modelos esperam um objeto do tipo `dgl.DGLGraph` como entrada.
Siga estas diretrizes:

1. **Estrutura Básica**
   ```python
   class NovaEstrategiaDataset(dgl.data.DGLDataset):
       def __init__(self, root, tsv_file, **kwargs):
           self.root = root
           self.data_type = "processed"
           self.save_path = osp.join(self.root, self.data_type)
           
           # Carrega dados do arquivo TSV
           with open(tsv_file, "r") as file:
               self.__train_data = file.readlines()
           
           # Processa labels
           self.labels = np.array([int(line.split("\t")[0]) 
                                 for line in self.__train_data])
           self.num_classes = len(np.unique(self.labels))
           
           # Inicializa listas para grafos e classes
           self.graph = []
           self.classes = []
           
           super().__init__(
               name="NovaEstrategia",
               raw_dir=root,
               save_dir=self.save_path,
               force_reload=False
           )

       def process(self):
           """
           Implementa a lógica de transformação série-grafo.
           Este método deve:
           1. Iterar sobre as séries temporais
           2. Converter cada série em um grafo
           3. Adicionar features aos nós/arestas
           4. Armazenar grafos e rótulos
           """
           for data in tqdm(self.__train_data, desc="Processing"):
               # Extrai label e sinal
               label = int(data.split("\t")[0])
               signal = np.array(data.split("\t")[1:]).astype(np.float32)
               
               # Implementa sua estratégia de conversão aqui
               # Exemplo: criar grafo baseado em alguma propriedade da série
               graph = create_graph(signal)  # método auxiliar
               
               # Adiciona features aos nós
               node_features = compute_node_features(graph, signal)
               graph.ndata['feat'] = node_features

               # Caso o grafo seja ponderado, adicionar:
               edge_weights = calcula_pesos(grafo) #lógica de cálculo de pesos
               graph.edata['weights'] = edge_weights
               
               # Armazena grafo e rótulo
               self.graph.append(graph)
               self.classes.append(torch.tensor(label))
           
           self.classes = torch.stack(self.classes)

   ```

2. **Boas Práticas**
   - Documente claramente a estratégia de transformação
   - Use métodos auxiliares para organizar o código
   - Implemente tratamento de erros robusto
   - Otimize operações pesadas usando NumPy/PyTorch
   - Mantenha consistência com outras estratégias do framework
   - Adicione testes unitários para sua implementação

3. **Exemplo de Uso**
   ```python
   # Registre sua estratégia em parameters.json
   {
       "valid_strategies": [
           "nova_estrategia",
           // ... outras estratégias ...
       ]
   }
   
   # Use na linha de comando
   python run.py --strategy nova_estrategia --model SAGE_MLPP_4layer
   ```

4. **Validação da Implementação**
   - Verifique se os grafos gerados são válidos
   - Teste com diferentes tamanhos de série temporal
   - Confirme que as features dos nós são apropriadas
   - Valide o processo de save/load
   - Compare resultados com outras estratégias

5. **Integração com o Framework**
   - Adicione sua estratégia ao módulo `datasets/`
   - Atualize a documentação com detalhes da estratégia
   - Forneça exemplos de uso e casos de teste
   - Mantenha compatibilidade com o pipeline existente

## Requisitos e Instalação

Para melhor eficiência, é recomendado utilizar GPUs para acelerar o treinamento.

### Dependências de Sistema

- **Docker**: Certifique-se de que o Docker está instalado e em execução. Você pode instalar o Docker seguindo as instruções oficiais [aqui](https://docs.docker.com/engine/install/).
- **NVIDIA Container Toolkit**: Necessário para suporte a GPU. Instale seguindo as instruções [aqui](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).   

### Construir e Executar a Imagem Docker

1. **Construir a Imagem Docker**:
   ```bash
   docker build -t graph-neural-alchemist .
   ```

2. **Executar o Container Docker**:
   ```bash
   docker run --gpus all -it --rm -v data/datasets:/data/datasets graph-neural-alchemist
   ```

   - `--gpus all`: Habilita o uso de todas as GPUs disponíveis.
   - `-it`: Abre um terminal interativo.
   - `--rm`: Remove o container após a execução.   
   - `-v data/datasets:/data/datasets`: Monta o diretório `data/datasets` no diretório `/data/datasets` dentro do container
3. **Verificar a Instalação**:
   Dentro do container, execute:
   ```bash
   python -c "import torch; import dgl; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'DGL versão: {dgl.__version__}')"
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
python run.py --strategy vg --model SAGE_MLPP_4layer --batch_size 32  --nhid 128 --epochs 250 --save_dir vg_sage_mlpp_4layer --root_path data/datasets
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
python run.py --strategy vg --model SAGE_MLPP_4layer --batch_size 32 --nhid 128 --epochs 250 --save_dir vg_sage_mlpp_4layer --root_path data/datasets --detect_anomaly --fast_dev_run_batches 10 --overfit_batches 10
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

Apache 2.0 License - veja [LICENSE](LICENSE) para detalhes.

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

## Configuração dos Experimentos

### Arquivo parameters.json

O framework utiliza um arquivo `parameters.json` para configurar os datasets e estratégias válidas. Este arquivo deve estar na raiz do projeto:

```json
{
    "datasets": [
        "Strawberry",
        "ECG200",
        "Coffee"
    ],
    "valid_strategies": [
        "op",
        "encoded_gtpo", 
        "vg",
        "simtsc",
        "pearson",
        "time2graphplus"
    ]
}
```

#### Campos do parameters.json:

- **datasets**: Lista de datasets a serem processados
  - Cada dataset deve ter seus arquivos TSV correspondentes em `{root_path}/{dataset}/`
  - Formato: `["{dataset1}", "{dataset2}", ...]`

- **valid_strategies**: Lista de estratégias de conversão série-grafo válidas
  - `"op"`: Grafos de Padrões Ordinais
  - `"encoded_gtpo"`: GTPO Codificado
  - `"vg"`: Grafos de Visibilidade
  - `"simtsc"`: SimTSC
  - `"pearson"`: Grafos de Covariância
  - `"time2graphplus"`: Time2Graph+

### Estrutura de Diretórios

```
projeto/
├── parameters.json
├── data/
│   ├── dataset1/
│   │   ├── dataset1_TRAIN.tsv
│   │   └── dataset1_TEST.tsv
│   └── dataset2/
│       ├── dataset2_TRAIN.tsv
│       └── dataset2_TEST.tsv
```

### Execução com parameters.json

1. **Configurar Datasets**:
   ```json
   {
       "datasets": ["Strawberry", "ECG200"]
   }
   ```

2. **Executar Experimento**:
   ```bash
   python run.py --strategy vg --model SAGE_MLPP_4layer
   ```
   - O script processará automaticamente todos os datasets listados em parameters.json

3. **Validação de Estratégias**:
   - O framework verifica se a estratégia fornecida está em `valid_strategies`
   - Se uma estratégia inválida for usada, um erro será registrado

### Exemplos de Uso

1. **Processamento Múltiplo**:
   ```json
   {
       "datasets": [
           "Strawberry",
           "ECG200",
           "Coffee"
       ]
   }
   ```
   ```bash
   python run.py --strategy vg --model SAGE_MLPP_4layer --batch_size 32
   ```

2. **Teste Rápido**:
   ```json
   {
       "datasets": ["Strawberry"]
   }
   ```
   ```bash
   python run.py --strategy vg --model SAGE_MLPP_4layer --fast_dev_run
   ```

3. **Experimento Completo**:
   ```json
   {
       "datasets": [
           "Strawberry",
           "ECG200",
           "Coffee",
           "GunPoint"
       ]
   }
   ```
   ```bash
   python run.py --strategy vg --model SAGE_MLPP_4layer --epochs 250 --early_stopping
   ```

### Logs e Monitoramento

- Os resultados são salvos em diretórios específicos para cada dataset
- Logs detalhados são gerados em `logs/YYYY-MM-DD_main.log`
- Métricas são visualizáveis via TensorBoard:
  ```bash
  tensorboard --logdir lightning_logs
  ```

### Dicas de Uso

1. **Desenvolvimento e Testes**:
   - Use um único dataset durante o desenvolvimento
   - Ative `--fast_dev_run` para testes rápidos
   - Use `--detect_anomaly` para debugging

2. **Experimentos Completos**:
   - Liste todos os datasets desejados em parameters.json
   - Use `--early_stopping` para otimizar o tempo de treinamento
   - Monitore via TensorBoard para acompanhar o progresso

3. **Customização**:
   - Adicione novas estratégias em `valid_strategies`
   - Organize datasets em subdiretórios apropriados
   - Mantenha a nomenclatura padrão dos arquivos TSV
