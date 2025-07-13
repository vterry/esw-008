package naivebayes

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/souza/esw-008/ml-nb-model/internal/models"
	"github.com/souza/esw-008/ml-nb-model/internal/utils"
)

// Classifier representa o classificador Naive Bayes
type Classifier struct {
	WordCounts  map[string]map[string]int
	ClassCounts map[string]int
	Vocab       map[string]bool
	StopWords   map[string]bool
}

// NewClassifier cria um novo classificador Naive Bayes
func NewClassifier() *Classifier {
	return &Classifier{
		WordCounts:  make(map[string]map[string]int),
		ClassCounts: make(map[string]int),
		Vocab:       make(map[string]bool),
		StopWords:   utils.GetStopWords(),
	}
}

// buildVocabularyNB constrói o vocabulário para Naive Bayes
func (c *Classifier) buildVocabularyNB(records []models.NewsRecord) {
	for _, record := range records {
		// Processar texto falso
		if strings.TrimSpace(record.FakeText) != "" {
			tokens := utils.PreprocessText(record.FakeText)
			for _, token := range tokens {
				c.Vocab[token] = true
			}
		}

		// Processar texto verdadeiro
		if strings.TrimSpace(record.TrueText) != "" {
			tokens := utils.PreprocessText(record.TrueText)
			for _, token := range tokens {
				c.Vocab[token] = true
			}
		}
	}
}

// TrainNB treina o classificador Naive Bayes
func (c *Classifier) TrainNB(records []models.NewsRecord) {
	// Construir vocabulário
	c.buildVocabularyNB(records)

	// Inicializar contadores
	c.WordCounts["true"] = make(map[string]int)
	c.WordCounts["fake"] = make(map[string]int)
	c.ClassCounts["true"] = 0
	c.ClassCounts["fake"] = 0

	// Contar palavras por classe
	for _, record := range records {
		// Processar texto falso
		if strings.TrimSpace(record.FakeText) != "" {
			c.ClassCounts["fake"]++
			tokens := utils.PreprocessText(record.FakeText)
			for _, token := range tokens {
				c.WordCounts["fake"][token]++
			}
		}

		// Processar texto verdadeiro
		if strings.TrimSpace(record.TrueText) != "" {
			c.ClassCounts["true"]++
			tokens := utils.PreprocessText(record.TrueText)
			for _, token := range tokens {
				c.WordCounts["true"][token]++
			}
		}
	}

	fmt.Println("Treinamento Naive Bayes concluído!")
}

// ClassifyNB classifica um texto usando Naive Bayes
func (c *Classifier) ClassifyNB(text string) (string, float64) {
	tokens := utils.PreprocessText(text)

	// Calcular log-probabilidades
	logProbTrue := math.Log(float64(c.ClassCounts["true"]))
	logProbFake := math.Log(float64(c.ClassCounts["fake"]))

	// Adicionar suavização de Laplace (α = 1)
	alpha := 1.0
	vocabSize := float64(len(c.Vocab))

	for _, token := range tokens {
		// Probabilidade para classe "true"
		countTrue := float64(c.WordCounts["true"][token])
		logProbTrue += math.Log((countTrue + alpha) / (float64(c.ClassCounts["true"]) + alpha*vocabSize))

		// Probabilidade para classe "fake"
		countFake := float64(c.WordCounts["fake"][token])
		logProbFake += math.Log((countFake + alpha) / (float64(c.ClassCounts["fake"]) + alpha*vocabSize))
	}

	// Determinar classe
	if logProbTrue > logProbFake {
		// Converter log-probabilidade para probabilidade
		prob := 1.0 / (1.0 + math.Exp(logProbFake-logProbTrue))
		return "true", prob * 100
	} else {
		prob := 1.0 / (1.0 + math.Exp(logProbTrue-logProbFake))
		return "fake", prob * 100
	}
}

// ClassifyWithDebugNB classifica um texto com informações detalhadas
func (c *Classifier) ClassifyWithDebugNB(text string) (string, float64, map[string]float64, []string) {
	tokens := utils.PreprocessText(text)

	// Calcular log-probabilidades
	logProbTrue := math.Log(float64(c.ClassCounts["true"]))
	logProbFake := math.Log(float64(c.ClassCounts["fake"]))

	// Adicionar suavização de Laplace (α = 1)
	alpha := 1.0
	vocabSize := float64(len(c.Vocab))

	// Calcular contribuição de cada token
	type tokenContribution struct {
		token string
		score float64
	}
	var contributions []tokenContribution

	for _, token := range tokens {
		// Probabilidade para classe "true"
		countTrue := float64(c.WordCounts["true"][token])
		probTrue := math.Log((countTrue + alpha) / (float64(c.ClassCounts["true"]) + alpha*vocabSize))
		logProbTrue += probTrue

		// Probabilidade para classe "fake"
		countFake := float64(c.WordCounts["fake"][token])
		probFake := math.Log((countFake + alpha) / (float64(c.ClassCounts["fake"]) + alpha*vocabSize))
		logProbFake += probFake

		// Contribuição do token (diferença entre as probabilidades)
		contribution := probFake - probTrue // Negativo = mais verdadeiro, Positivo = mais falso
		contributions = append(contributions, tokenContribution{token, contribution})
	}

	// Ordenar por contribuição (mais influentes primeiro)
	sort.Slice(contributions, func(i, j int) bool {
		return math.Abs(contributions[i].score) > math.Abs(contributions[j].score)
	})

	// Extrair tokens mais influentes
	var topTokens []string
	for _, contrib := range contributions {
		topTokens = append(topTokens, contrib.token)
		if len(topTokens) >= 10 {
			break
		}
	}

	// Calcular probabilidades finais
	probTrue := 1.0 / (1.0 + math.Exp(logProbFake-logProbTrue))
	probFake := 1.0 - probTrue

	probs := map[string]float64{
		"true": probTrue * 100,
		"fake": probFake * 100,
	}

	// Determinar classe e confiança
	var label string
	var confidence float64
	if logProbTrue > logProbFake {
		label = "true"
		confidence = probTrue * 100
	} else {
		label = "fake"
		confidence = probFake * 100
	}

	return label, confidence, probs, topTokens
}
