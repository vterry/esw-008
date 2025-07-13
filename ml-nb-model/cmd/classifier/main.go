package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"

	"github.com/souza/esw-008/ml-nb-model/internal/crawler"
	"github.com/souza/esw-008/ml-nb-model/internal/mlp"
	"github.com/souza/esw-008/ml-nb-model/internal/models"
	"github.com/souza/esw-008/ml-nb-model/internal/naivebayes"
	"github.com/souza/esw-008/ml-nb-model/internal/utils"
)

// loadDatasetFromURL carrega o dataset de uma URL
func loadDatasetFromURL(url string) ([]models.NewsRecord, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return parseDataset(resp.Body)
}

// parseDataset parseia o dataset CSV
func parseDataset(r io.Reader) ([]models.NewsRecord, error) {
	reader := csv.NewReader(r)
	reader.Comma = ','

	// Pular cabeçalho
	_, err := reader.Read()
	if err != nil {
		return nil, err
	}

	var records []models.NewsRecord
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if len(record) >= 5 {
			newsRecord := models.NewsRecord{
				TitleFake: record[0],
				FakeText:  record[1],
				LinkFake:  record[2],
				TrueText:  record[3],
				LinkTrue:  record[4],
			}
			records = append(records, newsRecord)
		}
	}

	return records, nil
}

// createFolds cria os folds para cross-validation
func createFolds(records []models.NewsRecord, numFolds int) []models.Fold {
	folds := make([]models.Fold, numFolds)

	// Dividir registros em folds
	for i, record := range records {
		foldIndex := i % numFolds
		folds[foldIndex].Test = append(folds[foldIndex].Test, record)
	}

	// Para cada fold, usar os outros como treinamento
	for i := range folds {
		for j, fold := range folds {
			if j != i {
				folds[i].Train = append(folds[i].Train, fold.Test...)
			}
		}
	}

	return folds
}

// calculateMetrics calcula métricas de avaliação
func calculateMetrics(predictions, actuals []string) models.Metrics {
	var tp, fp, tn, fn int

	for i, pred := range predictions {
		actual := actuals[i]
		if pred == "true" && actual == "true" {
			tp++
		} else if pred == "true" && actual == "fake" {
			fp++
		} else if pred == "fake" && actual == "true" {
			fn++
		} else if pred == "fake" && actual == "fake" {
			tn++
		}
	}

	accuracy := float64(tp+tn) / float64(tp+tn+fp+fn)
	precision := 0.0
	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp)
	}
	recall := 0.0
	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn)
	}
	f1Score := 0.0
	if precision+recall > 0 {
		f1Score = 2 * (precision * recall) / (precision + recall)
	}

	return models.Metrics{
		Accuracy:  accuracy,
		Precision: precision,
		Recall:    recall,
		F1Score:   f1Score,
	}
}

// evaluateModel avalia um modelo usando cross-validation
func evaluateModel(records []models.NewsRecord, algorithm string) models.Metrics {
	fmt.Printf("Avaliando modelo %s com 5-fold cross-validation...\n", algorithm)

	folds := createFolds(records, 5)
	var totalTP, totalFP, totalFN, totalTN int

	for i, fold := range folds {
		fmt.Printf("Fold %d/%d\n", i+1, 5)

		var predictions []string
		var trueLabels []string

		// Preparar dados de teste
		for _, record := range fold.Test {
			// Para cada registro, testar tanto o texto falso quanto o verdadeiro
			if strings.TrimSpace(record.FakeText) != "" {
				trueLabels = append(trueLabels, "fake")
				if algorithm == "MLP" {
					classifier := mlp.NewClassifier(1000, 50, 2)
					// Reduzir épocas para cross-validation (mais rápido)
					classifier.Epochs = 10
					classifier.Train(fold.Train)
					pred, _ := classifier.Classify(record.FakeText)
					predictions = append(predictions, pred)
				} else {
					classifier := naivebayes.NewClassifier()
					classifier.TrainNB(fold.Train)
					pred, _ := classifier.ClassifyNB(record.FakeText)
					predictions = append(predictions, pred)
				}
			}

			if strings.TrimSpace(record.TrueText) != "" {
				trueLabels = append(trueLabels, "true")
				if algorithm == "MLP" {
					classifier := mlp.NewClassifier(1000, 50, 2)
					// Reduzir épocas para cross-validation (mais rápido)
					classifier.Epochs = 10
					classifier.Train(fold.Train)
					pred, _ := classifier.Classify(record.TrueText)
					predictions = append(predictions, pred)
				} else {
					classifier := naivebayes.NewClassifier()
					classifier.TrainNB(fold.Train)
					pred, _ := classifier.ClassifyNB(record.TrueText)
					predictions = append(predictions, pred)
				}
			}
		}

		// Calcular métricas para este fold
		for j, pred := range predictions {
			trueLabel := trueLabels[j]
			if pred == "true" && trueLabel == "true" {
				totalTP++
			} else if pred == "true" && trueLabel == "fake" {
				totalFP++
			} else if pred == "fake" && trueLabel == "true" {
				totalFN++
			} else if pred == "fake" && trueLabel == "fake" {
				totalTN++
			}
		}
	}

	// Calcular métricas finais
	accuracy := float64(totalTP+totalTN) / float64(totalTP+totalTN+totalFP+totalFN)
	precision := 0.0
	if totalTP+totalFP > 0 {
		precision = float64(totalTP) / float64(totalTP+totalFP)
	}
	recall := 0.0
	if totalTP+totalFN > 0 {
		recall = float64(totalTP) / float64(totalTP+totalFN)
	}
	f1Score := 0.0
	if precision+recall > 0 {
		f1Score = 2 * (precision * recall) / (precision + recall)
	}

	return models.Metrics{
		Accuracy:  accuracy,
		Precision: precision,
		Recall:    recall,
		F1Score:   f1Score,
	}
}

// classifyNews classifica uma notícia usando o algoritmo especificado
func classifyNews(url string, algorithm string) {
	fmt.Printf("Analisando a URL: %s\n", url)

	articleText, err := crawler.CrawlNews(url)
	if err != nil {
		log.Fatalf("Erro ao extrair o conteúdo da notícia: %v", err)
	}

	if strings.TrimSpace(articleText) == "" {
		fmt.Println("Não foi possível extrair texto relevante da página.")
		return
	}

	if len(articleText) < 300 {
		fmt.Printf("[AVISO] O texto extraído é muito pequeno (%d caracteres). O resultado pode não ser confiável.\n", len(articleText))
	}
	fmt.Printf("Texto extraído (%d caracteres): %s...\n\n", len(articleText), articleText[:utils.Min(200, len(articleText))])

	// Carregar dataset para treinamento
	datasetURL := "https://raw.githubusercontent.com/jpchav98/FakeTrue.Br/refs/heads/main/FakeTrueBr_corpus.csv"
	records, err := loadDatasetFromURL(datasetURL)
	if err != nil {
		log.Fatalf("Falha ao carregar o dataset: %v", err)
	}

	var label string
	var confidence float64
	var probs map[string]float64
	var topTokens []string

	if algorithm == "MLP" {
		fmt.Println("Treinando classificador MLP...")
		classifier := mlp.NewClassifier(1000, 50, 2)
		classifier.Train(records)
		label, confidence, probs, topTokens = classifier.ClassifyWithDebug(articleText)
	} else {
		fmt.Println("Treinando classificador Naive Bayes...")
		classifier := naivebayes.NewClassifier()
		classifier.TrainNB(records)
		label, confidence, probs, topTokens = classifier.ClassifyWithDebugNB(articleText)
	}

	// Map the label for better display
	var result string
	if label == "true" {
		result = "Provavelmente Verdadeira"
	} else {
		result = "Provavelmente Falsa"
	}

	lowerText := strings.ToLower(articleText)
	if strings.Contains(lowerText, "boato") || strings.Contains(lowerText, "falso") ||
		strings.Contains(lowerText, "mentira") || strings.Contains(lowerText, "desmentido") ||
		strings.Contains(lowerText, "fake news") {
		fmt.Println("[HEURÍSTICA] O texto contém termos típicos de desmentido ou fake news.")
		fmt.Println("--- Resultado da Análise ---")
		fmt.Println("Classificação: Provavelmente Falsa (por heurística)")
		fmt.Println("----------------------------")
		return
	}

	fmt.Println("--- Resultado da Análise ---")
	fmt.Printf("Algoritmo utilizado: %s\n", algorithm)
	fmt.Printf("Classificação: %s\n", result)
	fmt.Printf("Confiança: %.2f%%\n", confidence)
	fmt.Printf("Probabilidades: Verdadeira: %.2f%% | Falsa: %.2f%%\n", probs["true"], probs["fake"])
	fmt.Printf("Tokens mais influentes para a decisão: %v\n", topTokens)
	fmt.Printf("URL analisada: %s\n", url)
	fmt.Println("----------------------------")
}

// compareAlgorithms compara MLP e Naive Bayes em uma URL específica
func compareAlgorithms(url string, records []models.NewsRecord) {
	fmt.Printf("Analisando a URL: %s\n", url)

	articleText, err := crawler.CrawlNews(url)
	if err != nil {
		log.Fatalf("Erro ao extrair o conteúdo da notícia: %v", err)
	}

	if strings.TrimSpace(articleText) == "" {
		fmt.Println("Não foi possível extrair texto relevante da página.")
		return
	}

	if len(articleText) < 300 {
		fmt.Printf("[AVISO] O texto extraído é muito pequeno (%d caracteres). O resultado pode não ser confiável.\n", len(articleText))
	}
	fmt.Printf("Texto extraído (%d caracteres): %s...\n\n", len(articleText), articleText[:utils.Min(200, len(articleText))])

	// Verificar heurística
	lowerText := strings.ToLower(articleText)
	if strings.Contains(lowerText, "boato") || strings.Contains(lowerText, "falso") ||
		strings.Contains(lowerText, "mentira") || strings.Contains(lowerText, "desmentido") ||
		strings.Contains(lowerText, "fake news") {
		fmt.Println("[HEURÍSTICA] O texto contém termos típicos de desmentido ou fake news.")
		fmt.Println("--- Resultado da Análise ---")
		fmt.Println("Classificação: Provavelmente Falsa (por heurística)")
		fmt.Println("----------------------------")
		return
	}

	// Calcular métricas de cross-validation primeiro
	fmt.Println("=== CALCULANDO MÉTRICAS DE PERFORMANCE ===")
	fmt.Println("Executando 5-fold cross-validation...")
	fmt.Println("(Isso pode levar alguns minutos devido ao treinamento do MLP)")

	mlpMetrics := evaluateModel(records, "MLP")
	nbMetrics := evaluateModel(records, "Naive Bayes")

	// Treinar e testar MLP
	fmt.Println("\n=== ANÁLISE COM MLP ===")
	mlpClassifier := mlp.NewClassifier(1000, 50, 2)
	mlpClassifier.Train(records)
	mlpLabel, mlpConfidence, mlpProbs, mlpTokens := mlpClassifier.ClassifyWithDebug(articleText)

	// Treinar e testar Naive Bayes
	fmt.Println("\n=== ANÁLISE COM NAIVE BAYES ===")
	nbClassifier := naivebayes.NewClassifier()
	nbClassifier.TrainNB(records)
	nbLabel, nbConfidence, nbProbs, nbTokens := nbClassifier.ClassifyWithDebugNB(articleText)

	// Mapear labels para exibição
	var mlpResult, nbResult string
	if mlpLabel == "true" {
		mlpResult = "Provavelmente Verdadeira"
	} else {
		mlpResult = "Provavelmente Falsa"
	}

	if nbLabel == "true" {
		nbResult = "Provavelmente Verdadeira"
	} else {
		nbResult = "Provavelmente Falsa"
	}

	// Imprimir comparação
	fmt.Println("\n" + strings.Repeat("=", 120))
	fmt.Println("COMPARAÇÃO ENTRE ALGORITMOS")
	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("URL analisada: %s\n\n", url)

	// Tabela de resultados com métricas de cross-validation
	fmt.Printf("%-20s %-25s %-15s %-20s %-12s %-12s %-12s %-12s\n",
		"Algoritmo", "Classificação", "Confiança", "Probabilidades", "Acurácia", "Precisão", "Revocação", "F1-Score")
	fmt.Println(strings.Repeat("-", 120))
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s %-12.4f %-12.4f %-12.4f %-12.4f\n",
		"MLP", mlpResult, mlpConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", mlpProbs["true"], mlpProbs["fake"]),
		mlpMetrics.Accuracy, mlpMetrics.Precision, mlpMetrics.Recall, mlpMetrics.F1Score)
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s %-12.4f %-12.4f %-12.4f %-12.4f\n",
		"Naive Bayes", nbResult, nbConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", nbProbs["true"], nbProbs["fake"]),
		nbMetrics.Accuracy, nbMetrics.Precision, nbMetrics.Recall, nbMetrics.F1Score)
	fmt.Println(strings.Repeat("=", 120))

	// Detalhes dos tokens influentes
	fmt.Println("\n=== TOKENS MAIS INFLUENTES ===")
	fmt.Println("MLP:")
	for i, token := range mlpTokens {
		if i < 5 {
			fmt.Printf("  %s\n", token)
		}
	}
	fmt.Println("\nNaive Bayes:")
	for i, token := range nbTokens {
		if i < 5 {
			fmt.Printf("  %s\n", token)
		}
	}

	// Análise de concordância
	fmt.Println("\n=== ANÁLISE DE CONCORDÂNCIA ===")
	if mlpLabel == nbLabel {
		fmt.Printf("✅ Os algoritmos concordam: %s\n", mlpResult)
	} else {
		fmt.Printf("❌ Os algoritmos discordam:\n")
		fmt.Printf("   MLP: %s (%.2f%%)\n", mlpResult, mlpConfidence)
		fmt.Printf("   NB:  %s (%.2f%%)\n", nbResult, nbConfidence)
	}

	// Diferença de confiança
	confidenceDiff := math.Abs(mlpConfidence - nbConfidence)
	fmt.Printf("\nDiferença de confiança: %.2f%%\n", confidenceDiff)

	if confidenceDiff < 10 {
		fmt.Println("📊 Baixa divergência entre os algoritmos")
	} else if confidenceDiff < 25 {
		fmt.Println("📊 Divergência moderada entre os algoritmos")
	} else {
		fmt.Println("📊 Alta divergência entre os algoritmos")
	}

	// Comparação de performance geral
	fmt.Println("\n=== COMPARAÇÃO DE PERFORMANCE GERAL ===")
	fmt.Printf("MLP vs Naive Bayes:\n")
	fmt.Printf("  Acurácia:   %.4f vs %.4f (diferença: %.4f)\n",
		mlpMetrics.Accuracy, nbMetrics.Accuracy, mlpMetrics.Accuracy-nbMetrics.Accuracy)
	fmt.Printf("  Precisão:   %.4f vs %.4f (diferença: %.4f)\n",
		mlpMetrics.Precision, nbMetrics.Precision, mlpMetrics.Precision-nbMetrics.Precision)
	fmt.Printf("  Revocação:  %.4f vs %.4f (diferença: %.4f)\n",
		mlpMetrics.Recall, nbMetrics.Recall, mlpMetrics.Recall-nbMetrics.Recall)
	fmt.Printf("  F1-Score:   %.4f vs %.4f (diferença: %.4f)\n",
		mlpMetrics.F1Score, nbMetrics.F1Score, mlpMetrics.F1Score-nbMetrics.F1Score)

	fmt.Println(strings.Repeat("=", 120))
}

// compareAlgorithmsFast compara MLP e Naive Bayes em uma URL específica (versão rápida sem cross-validation)
func compareAlgorithmsFast(url string, records []models.NewsRecord) {
	fmt.Printf("Analisando a URL: %s\n", url)

	articleText, err := crawler.CrawlNews(url)
	if err != nil {
		log.Fatalf("Erro ao extrair o conteúdo da notícia: %v", err)
	}

	if strings.TrimSpace(articleText) == "" {
		fmt.Println("Não foi possível extrair texto relevante da página.")
		return
	}

	if len(articleText) < 300 {
		fmt.Printf("[AVISO] O texto extraído é muito pequeno (%d caracteres). O resultado pode não ser confiável.\n", len(articleText))
	}
	fmt.Printf("Texto extraído (%d caracteres): %s...\n\n", len(articleText), articleText[:utils.Min(200, len(articleText))])

	// Verificar heurística
	lowerText := strings.ToLower(articleText)
	if strings.Contains(lowerText, "boato") || strings.Contains(lowerText, "falso") ||
		strings.Contains(lowerText, "mentira") || strings.Contains(lowerText, "desmentido") ||
		strings.Contains(lowerText, "fake news") {
		fmt.Println("[HEURÍSTICA] O texto contém termos típicos de desmentido ou fake news.")
		fmt.Println("--- Resultado da Análise ---")
		fmt.Println("Classificação: Provavelmente Falsa (por heurística)")
		fmt.Println("----------------------------")
		return
	}

	// Treinar e testar MLP
	fmt.Println("=== ANÁLISE COM MLP ===")
	mlpClassifier := mlp.NewClassifier(1000, 50, 2)
	mlpClassifier.Train(records)
	mlpLabel, mlpConfidence, mlpProbs, mlpTokens := mlpClassifier.ClassifyWithDebug(articleText)

	// Treinar e testar Naive Bayes
	fmt.Println("\n=== ANÁLISE COM NAIVE BAYES ===")
	nbClassifier := naivebayes.NewClassifier()
	nbClassifier.TrainNB(records)
	nbLabel, nbConfidence, nbProbs, nbTokens := nbClassifier.ClassifyWithDebugNB(articleText)

	// Mapear labels para exibição
	var mlpResult, nbResult string
	if mlpLabel == "true" {
		mlpResult = "Provavelmente Verdadeira"
	} else {
		mlpResult = "Provavelmente Falsa"
	}

	if nbLabel == "true" {
		nbResult = "Provavelmente Verdadeira"
	} else {
		nbResult = "Provavelmente Falsa"
	}

	// Imprimir comparação
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("COMPARAÇÃO ENTRE ALGORITMOS (VERSÃO RÁPIDA)")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("URL analisada: %s\n\n", url)

	// Tabela de resultados (sem métricas de cross-validation)
	fmt.Printf("%-20s %-25s %-15s %-20s\n", "Algoritmo", "Classificação", "Confiança", "Probabilidades")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s\n", "MLP", mlpResult, mlpConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", mlpProbs["true"], mlpProbs["fake"]))
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s\n", "Naive Bayes", nbResult, nbConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", nbProbs["true"], nbProbs["fake"]))
	fmt.Println(strings.Repeat("=", 80))

	// Detalhes dos tokens influentes
	fmt.Println("\n=== TOKENS MAIS INFLUENTES ===")
	fmt.Println("MLP:")
	for i, token := range mlpTokens {
		if i < 5 {
			fmt.Printf("  %s\n", token)
		}
	}
	fmt.Println("\nNaive Bayes:")
	for i, token := range nbTokens {
		if i < 5 {
			fmt.Printf("  %s\n", token)
		}
	}

	// Análise de concordância
	fmt.Println("\n=== ANÁLISE DE CONCORDÂNCIA ===")
	if mlpLabel == nbLabel {
		fmt.Printf("✅ Os algoritmos concordam: %s\n", mlpResult)
	} else {
		fmt.Printf("❌ Os algoritmos discordam:\n")
		fmt.Printf("   MLP: %s (%.2f%%)\n", mlpResult, mlpConfidence)
		fmt.Printf("   NB:  %s (%.2f%%)\n", nbResult, nbConfidence)
	}

	// Diferença de confiança
	confidenceDiff := math.Abs(mlpConfidence - nbConfidence)
	fmt.Printf("\nDiferença de confiança: %.2f%%\n", confidenceDiff)

	if confidenceDiff < 10 {
		fmt.Println("📊 Baixa divergência entre os algoritmos")
	} else if confidenceDiff < 25 {
		fmt.Println("📊 Divergência moderada entre os algoritmos")
	} else {
		fmt.Println("📊 Alta divergência entre os algoritmos")
	}

	fmt.Println(strings.Repeat("=", 80))
}

// main é o ponto de entrada da aplicação
func main() {
	if len(os.Args) < 2 {
		fmt.Println("Uso:")
		fmt.Println("  go run cmd/classifier/main.go <url>                     # Analisa URL com MLP e NB")
		fmt.Println("  go run cmd/classifier/main.go mlp <url>                 # Usa apenas MLP")
		fmt.Println("  go run cmd/classifier/main.go nb <url>                  # Usa apenas Naive Bayes")
		fmt.Println("  go run cmd/classifier/main.go fast <url>                # Comparação rápida (sem cross-validation)")
		fmt.Println("")
		fmt.Println("Exemplos:")
		fmt.Println("  go run cmd/classifier/main.go https://g1.globo.com/...")
		fmt.Println("  go run cmd/classifier/main.go mlp https://g1.globo.com/...")
		fmt.Println("  go run cmd/classifier/main.go nb https://g1.globo.com/...")
		fmt.Println("  go run cmd/classifier/main.go fast https://g1.globo.com/...")
		return
	}

	// Carregar dataset
	datasetURL := "https://raw.githubusercontent.com/jpchav98/FakeTrue.Br/refs/heads/main/FakeTrueBr_corpus.csv"
	records, err := loadDatasetFromURL(datasetURL)
	if err != nil {
		log.Fatalf("Falha ao carregar o dataset: %v", err)
	}

	if os.Args[1] == "mlp" {
		if len(os.Args) < 3 {
			fmt.Println("Erro: URL necessária para classificação com MLP")
			fmt.Println("Uso: go run cmd/classifier/main.go mlp <url>")
			return
		}
		classifyNews(os.Args[2], "MLP")

	} else if os.Args[1] == "nb" {
		if len(os.Args) < 3 {
			fmt.Println("Erro: URL necessária para classificação com Naive Bayes")
			fmt.Println("Uso: go run cmd/classifier/main.go nb <url>")
			return
		}
		classifyNews(os.Args[2], "Naive Bayes")

	} else if os.Args[1] == "fast" {
		if len(os.Args) < 3 {
			fmt.Println("Erro: URL necessária para comparação rápida")
			fmt.Println("Uso: go run cmd/classifier/main.go fast <url>")
			return
		}
		compareAlgorithmsFast(os.Args[2], records)

	} else {
		// Comportamento padrão: comparar MLP e NB na URL fornecida
		compareAlgorithms(os.Args[1], records)
	}
}
