# Exemplo de Teste - Classificador de Fake News

Este arquivo demonstra como usar o sistema de classificação de fake news com comparação entre algoritmos.

## Como Executar

### 1. Compilar o Programa
```bash
go build -o classifier main.go
```

### 2. Testar com uma URL

#### Exemplo 1: Comparação de Algoritmos (Padrão)
```bash
./classifier "https://www.uol.com.br/"
```

**Saída esperada:**
```
Analisando a URL: https://www.uol.com.br/
Texto extraído (1650 caracteres): Time inglês vence favorito PSG por 3 a 0...

=== ANÁLISE COM MLP ===
Construindo vocabulário...
Vocabulário construído com 1000 palavras
Dados de treinamento preparados: 3582 amostras
Iniciando treinamento da rede neural...
Época 0/100, Erro: 791.915837
...
Treinamento concluído!

=== ANÁLISE COM NAIVE BAYES ===
Construindo vocabulário para Naive Bayes...
Vocabulário construído com 45803 palavras
Treinamento Naive Bayes concluído!

================================================================================
COMPARAÇÃO ENTRE ALGORITMOS
================================================================================
URL analisada: https://www.uol.com.br/

Algoritmo            Classificação             Confiança       Probabilidades      
--------------------------------------------------------------------------------
MLP                  Provavelmente Falsa       99.49%         V:0.5% F:99.5%      
Naive Bayes          Provavelmente Falsa       100.00%        V:0.0% F:100.0%     
================================================================================

=== TOKENS MAIS INFLUENTES ===
MLP:
  time (1.00)
  inglês (1.00)
  vence (1.00)
  favorito (1.00)
  psg (1.00)

Naive Bayes:
  psg (-60.99)
  chelsea (-36.60)
  trump (-34.46)
  clubes (-24.40)
  favorito (-22.20)

=== ANÁLISE DE CONCORDÂNCIA ===
✅ Os algoritmos concordam: Provavelmente Falsa

Diferença de confiança: 0.51%
📊 Baixa divergência entre os algoritmos
================================================================================
```

#### Exemplo 2: Usar Apenas MLP
```bash
./classifier mlp "https://www.uol.com.br/"
```

#### Exemplo 3: Usar Apenas Naive Bayes
```bash
./classifier nb "https://www.uol.com.br/"
```

## Interpretação dos Resultados

### Classificação
- **Provavelmente Verdadeira**: O algoritmo considera a notícia como verdadeira
- **Provavelmente Falsa**: O algoritmo considera a notícia como falsa

### Confiança
- **0-50%**: Baixa confiança na classificação
- **50-75%**: Confiança moderada
- **75-90%**: Alta confiança
- **90-100%**: Muito alta confiança

### Probabilidades
- **V**: Probabilidade de ser verdadeira
- **F**: Probabilidade de ser falsa
- A soma sempre é 100%

### Tokens Mais Influentes
- **MLP**: Palavras com maior peso no vetor de entrada (valores positivos)
- **Naive Bayes**: Palavras com maior contribuição logarítmica (valores negativos = mais falsas)

### Análise de Concordância
- **✅ Concordância**: Ambos os algoritmos chegam à mesma conclusão
- **❌ Discordância**: Algoritmos chegam a conclusões diferentes

### Níveis de Divergência
- **Baixa**: < 10% de diferença de confiança
- **Moderada**: 10-25% de diferença
- **Alta**: > 25% de diferença

## URLs de Teste Sugeridas

### Sites de Notícias Confiáveis
- https://g1.globo.com/
- https://www.uol.com.br/
- https://www.estadao.com.br/
- https://www.folha.uol.com.br/

### Sites de Verificação de Fatos
- https://www.boatos.org/
- https://www.e-farsas.com/
- https://www.lupa.uol.com.br/

## Notas Importantes

1. **Tempo de Processamento**: O MLP pode levar alguns minutos para treinar
2. **Qualidade do Texto**: URLs com pouco texto podem gerar resultados menos confiáveis
3. **Heurísticas**: O sistema detecta automaticamente termos como "boato", "falso", etc.
4. **Vocabulário**: MLP usa 1000 palavras mais frequentes, NB usa todo o vocabulário
5. **Memória**: NB é mais eficiente em memória que MLP

## Troubleshooting

### Erro 404
```
Erro ao extrair o conteúdo da notícia: falha na requisição: status code 404
```
**Solução**: Use uma URL válida de notícia

### Texto Muito Pequeno
```
[AVISO] O texto extraído é muito pequeno (150 caracteres). O resultado pode não ser confiável.
```
**Solução**: Use URLs de artigos completos, não páginas principais

### Timeout
```
Teste concluído (timeout após 120s)
```
**Solução**: O MLP pode demorar para treinar. Aguarde ou use apenas NB com `./classifier nb <URL>` 