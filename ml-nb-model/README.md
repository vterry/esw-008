# Classificador de Fake News com MLP e Naive Bayes

Este projeto implementa classificadores de fake news usando MLP (Multi-Layer Perceptron) e Naive Bayes em Go. O sistema analisa textos de notícias e classifica se são provavelmente verdadeiras ou falsas, com suporte a comparação entre algoritmos em uma URL específica.

## Estrutura do Projeto

```
ml-nb-model/
├── cmd/
│   └── classifier/
│       └── main.go              # Ponto de entrada principal
├── internal/
│   ├── models/
│   │   └── types.go             # Estruturas de dados
│   ├── utils/
│   │   └── text_processing.go   # Processamento de texto
│   ├── crawler/
│   │   └── web_crawler.go       # Web scraping
│   ├── mlp/
│   │   └── classifier.go        # Classificador MLP
│   └── naivebayes/
│       └── classifier.go        # Classificador Naive Bayes
├── test_urls_analysis.go        # Teste das 5 URLs especificadas
├── main.go                      # Arquivo original (legado)
├── go.mod                       # Dependências do projeto
└── README.md                    # Este arquivo
```

## Características

- **Dois Algoritmos**: MLP (rede neural) e Naive Bayes
- **Comparação em Tempo Real**: Analisa uma URL com ambos os algoritmos
- **Métricas Detalhadas**: Confiança, probabilidades e tokens influentes
- **Processamento de Texto**: Tokenização e remoção de stop words em português
- **Vocabulário Dinâmico**: Construído automaticamente a partir dos dados de treinamento
- **Web Scraping**: Extração automática de conteúdo de URLs de notícias
- **Classificação Binária**: Verdadeira vs Falsa
- **Arquitetura Modular**: Separação clara de responsabilidades

## Arquitetura dos Algoritmos

### MLP (Multi-Layer Perceptron)
```
Entrada (1000) → Camada Oculta (50) → Saída (2)
```

- **Camada de Entrada**: 1000 neurônios (tamanho do vocabulário)
- **Camada Oculta**: 50 neurônios com função de ativação sigmoid
- **Camada de Saída**: 2 neurônios (verdadeira/falsa) com função de ativação sigmoid

### Naive Bayes
- **Probabilístico**: Baseado em teorema de Bayes
- **Suavização de Laplace**: Para lidar com palavras não vistas
- **Log-probabilidades**: Para estabilidade numérica

## Parâmetros de Treinamento

### MLP
- **Learning Rate**: 0.01
- **Épocas**: 100 (padrão) / 10 (cross-validation)
- **Função de Ativação**: Sigmoid
- **Função de Perda**: Mean Squared Error (MSE)

### Naive Bayes
- **Suavização**: Laplace (α = 1)
- **Vocabulário**: Todas as palavras únicas
- **Stop Words**: Removidas automaticamente

## Como Usar

### Compilação

```bash
# Compilar o classificador principal
go build -o classifier cmd/classifier/main.go

# Compilar o teste das URLs
go build -o test_analysis test_urls_analysis.go
```

### Execução

#### 1. Comparação de Algoritmos (Padrão - com Cross-Validation)
```bash
./classifier <URL_da_noticia>
```
ou
```bash
go run cmd/classifier/main.go <URL_da_noticia>
```

**Saída esperada:**
```
========================================================================================================================
COMPARAÇÃO ENTRE ALGORITMOS
========================================================================================================================
URL analisada: https://exemplo.com/noticia

Algoritmo           Classificação           Confiança        Probabilidades      Acurácia     Precisão     Revocação    F1-Score
------------------------------------------------------------------------------------------------------------------------
MLP                 Provavelmente Verdadeira 85.32%         V:85.3% F:14.7%     0.8500       0.8600       0.8400       0.8500
Naive Bayes         Provavelmente Verdadeira 82.15%         V:82.2% F:17.8%     0.8200       0.8300       0.8100       0.8200
========================================================================================================================

=== COMPARAÇÃO DE PERFORMANCE GERAL ===
MLP vs Naive Bayes:
  Acurácia:   0.8500 vs 0.8200 (diferença: 0.0300)
  Precisão:   0.8600 vs 0.8300 (diferença: 0.0300)
  Revocação:  0.8400 vs 0.8100 (diferença: 0.0300)
  F1-Score:   0.8500 vs 0.8200 (diferença: 0.0300)
========================================================================================================================
```

#### 2. Comparação Rápida (Sem Cross-Validation)
```bash
./classifier fast <URL_da_noticia>
```
ou
```bash
go run cmd/classifier/main.go fast <URL_da_noticia>
```

**Saída esperada:**
```
================================================================================
COMPARAÇÃO ENTRE ALGORITMOS (VERSÃO RÁPIDA)
================================================================================
URL analisada: https://exemplo.com/noticia

Algoritmo           Classificação           Confiança        Probabilidades
--------------------------------------------------------------------------------
MLP                 Provavelmente Verdadeira 85.32%         V:85.3% F:14.7%
Naive Bayes         Provavelmente Verdadeira 82.15%         V:82.2% F:17.8%
================================================================================

=== TOKENS MAIS INFLUENTES ===
MLP:
  palavra1 (1.00)
  palavra2 (1.00)
  ...

Naive Bayes:
  palavra1 (-2.34)
  palavra2 (-1.87)
  ...

=== ANÁLISE DE CONCORDÂNCIA ===
✅ Os algoritmos concordam: Provavelmente Verdadeira

Diferença de confiança: 3.17%
📊 Baixa divergência entre os algoritmos
================================================================================
```

#### 3. Classificação com MLP Apenas
```bash
./classifier mlp <URL_da_noticia>
```
ou
```bash
go run cmd/classifier/main.go mlp <URL_da_noticia>
```

#### 4. Classificação com Naive Bayes Apenas
```bash
./classifier nb <URL_da_noticia>
```
ou
```bash
go run cmd/classifier/main.go nb <URL_da_noticia>
```

### Exemplos de Uso

```bash
# Comparação completa com cross-validation (mais lento)
./classifier https://g1.globo.com/noticia-exemplo

# Comparação rápida sem cross-validation
./classifier fast https://g1.globo.com/noticia-exemplo

# Usar apenas MLP
./classifier mlp https://g1.globo.com/noticia-exemplo

# Usar apenas Naive Bayes
./classifier nb https://g1.globo.com/noticia-exemplo
```

## Teste das 5 URLs Especificadas

Para testar as 5 URLs especificadas:

```bash
# Executar o teste
./test_analysis

# Ou executar diretamente
go run test_urls_analysis.go
```

**URLs testadas:**
1. **G1 - Saúde**: Higiene do sono
2. **G1 - Política**: Denúncia MPMG sobre drone
3. **Boatos.org - Política**: Boato sobre Haddad e imposto do oxigênio
4. **Estadão - Verificação**: Contas dark no Instagram
5. **Boatos.org - Política**: Boato sobre porta-aviões no Lago Paranoá

## Saída de Comparação

O sistema fornece informações detalhadas sobre a classificação de ambos os algoritmos:

### Tabela de Resultados (Versão Completa)
- **Algoritmo**: MLP ou Naive Bayes
- **Classificação**: "Provavelmente Verdadeira" ou "Provavelmente Falsa"
- **Confiança**: Percentual de confiança da classificação
- **Probabilidades**: Probabilidades para cada classe (Verdadeira/Falsa)
- **Acurácia**: Taxa de acertos gerais (5-fold cross-validation)
- **Precisão**: Taxa de verdadeiros positivos entre os preditos como positivos
- **Revocação**: Taxa de verdadeiros positivos entre os reais positivos
- **F1-Score**: Média harmônica entre precisão e revocação

### Tabela de Resultados (Versão Rápida)
- **Algoritmo**: MLP ou Naive Bayes
- **Classificação**: "Provavelmente Verdadeira" ou "Provavelmente Falsa"
- **Confiança**: Percentual de confiança da classificação
- **Probabilidades**: Probabilidades para cada classe (Verdadeira/Falsa)

### Tokens Mais Influentes
- **MLP**: Palavras com maior peso no vetor de entrada (valores positivos)
- **Naive Bayes**: Palavras com maior contribuição logarítmica (valores negativos = mais falsas)

### Análise de Concordância
- **✅ Concordância**: Ambos os algoritmos chegam à mesma conclusão
- **❌ Discordância**: Algoritmos chegam a conclusões diferentes
- **Diferença de Confiança**: Medida da divergência entre os algoritmos

### Níveis de Divergência
- **Baixa**: < 10% de diferença
- **Moderada**: 10-25% de diferença
- **Alta**: > 25% de diferença

### Comparação de Performance Geral (Versão Completa)
- **Acurácia**: Comparação direta entre os algoritmos
- **Precisão**: Qualidade das predições positivas
- **Revocação**: Capacidade de encontrar todos os positivos
- **F1-Score**: Balanceamento entre precisão e revocação

## Dataset

O sistema utiliza o dataset FakeTrue.Br, que contém pares de notícias verdadeiras e falsas em português brasileiro. O dataset é baixado automaticamente durante a execução.

## Processamento de Texto

1. **Tokenização**: Divisão do texto em palavras
2. **Normalização**: Conversão para minúsculas
3. **Limpeza**: Remoção de pontuação e caracteres especiais
4. **Filtragem**: Remoção de stop words em português
5. **Vectorização**: Conversão para vetor de entrada (MLP) ou contagem de palavras (NB)

## Comparação entre Algoritmos

### Vantagens do MLP:
1. **Capacidade de Aprendizado Não-Linear**: Pode capturar relações complexas entre palavras
2. **Representação Distribuída**: Palavras similares podem ter representações similares
3. **Flexibilidade**: Pode ser facilmente expandido com mais camadas ou neurônios

### Vantagens do Naive Bayes:
1. **Velocidade**: Treinamento e classificação muito rápidos
2. **Interpretabilidade**: Fácil de entender como as decisões são tomadas
3. **Eficiência**: Requer menos dados para treinamento
4. **Estabilidade**: Menos propenso a overfitting

## Estrutura do Código

### Módulos Principais:
- `cmd/classifier/main.go`: Ponto de entrada principal
- `internal/models/types.go`: Estruturas de dados
- `internal/utils/text_processing.go`: Processamento de texto
- `internal/crawler/web_crawler.go`: Web scraping
- `internal/mlp/classifier.go`: Classificador MLP
- `internal/naivebayes/classifier.go`: Classificador Naive Bayes

### Funções Principais:
- `compareAlgorithms`: Comparação completa com cross-validation
- `compareAlgorithmsFast`: Comparação rápida sem cross-validation
- `classifyNews`: Classificação com algoritmo específico
- `evaluateModel`: Avaliação com cross-validation
- `analyzeURLTest`: Análise de URL específica (teste)

## Dependências

- `github.com/PuerkitoBio/goquery`: Para web scraping
- Bibliotecas padrão do Go: `math`, `math/rand`, `strings`, `regexp`, etc.

## Limitações

1. **Vocabulário Fixo**: MLP limitado a 1000 palavras mais frequentes
2. **Arquitetura Simples**: MLP com apenas uma camada oculta
3. **Tempo de Processamento**: MLP requer mais tempo que Naive Bayes
4. **Memória**: MLP requer mais memória que Naive Bayes

## Possíveis Melhorias

1. **Embeddings**: Usar word embeddings ao invés de one-hot encoding
2. **Arquitetura Mais Profunda**: Adicionar mais camadas ocultas ao MLP
3. **Dropout**: Implementar dropout para regularização
4. **Batch Training**: Treinar em lotes para melhor performance
5. **Early Stopping**: Parar treinamento quando erro não diminui
6. **Hiperparâmetros**: Otimização automática de hiperparâmetros
7. **Ensemble Methods**: Combinação dos dois algoritmos 