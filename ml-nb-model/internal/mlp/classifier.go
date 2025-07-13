package mlp

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/souza/esw-008/ml-nb-model/internal/models"
	"github.com/souza/esw-008/ml-nb-model/internal/utils"
)

// Neuron representa um neurônio da rede neural
type Neuron struct {
	Weights []float64
	Bias    float64
	Output  float64
	Delta   float64
}

// Layer representa uma camada da rede neural
type Layer struct {
	Neurons []*Neuron
}

// Classifier representa o classificador MLP
type Classifier struct {
	Layers       []*Layer
	Vocab        map[string]int
	StopWords    map[string]bool
	InputSize    int
	HiddenSize   int
	OutputSize   int
	LearningRate float64
	Epochs       int
}

// NewClassifier cria um novo classificador MLP
func NewClassifier(inputSize, hiddenSize, outputSize int) *Classifier {
	classifier := &Classifier{
		InputSize:    inputSize,
		HiddenSize:   hiddenSize,
		OutputSize:   outputSize,
		LearningRate: 0.01,
		Epochs:       100,
		Vocab:        make(map[string]int),
		StopWords:    utils.GetStopWords(),
	}

	classifier.initializeLayers()
	return classifier
}

// initializeLayers inicializa as camadas da rede neural
func (c *Classifier) initializeLayers() {
	// Camada de entrada (não tem neurônios, apenas passa os dados)

	// Camada oculta
	hiddenLayer := &Layer{}
	for i := 0; i < c.HiddenSize; i++ {
		neuron := &Neuron{
			Weights: make([]float64, c.InputSize),
			Bias:    rand.Float64()*0.2 - 0.1,
		}
		for j := 0; j < c.InputSize; j++ {
			neuron.Weights[j] = rand.Float64()*0.2 - 0.1
		}
		hiddenLayer.Neurons = append(hiddenLayer.Neurons, neuron)
	}

	// Camada de saída
	outputLayer := &Layer{}
	for i := 0; i < c.OutputSize; i++ {
		neuron := &Neuron{
			Weights: make([]float64, c.HiddenSize),
			Bias:    rand.Float64()*0.2 - 0.1,
		}
		for j := 0; j < c.HiddenSize; j++ {
			neuron.Weights[j] = rand.Float64()*0.2 - 0.1
		}
		outputLayer.Neurons = append(outputLayer.Neurons, neuron)
	}

	c.Layers = []*Layer{hiddenLayer, outputLayer}
}

// buildVocabulary constrói o vocabulário a partir dos dados de treinamento
func (c *Classifier) buildVocabulary(records []models.NewsRecord) {
	wordCounts := make(map[string]int)

	// Contar frequência de palavras
	for _, record := range records {
		// Processar texto falso
		if strings.TrimSpace(record.FakeText) != "" {
			tokens := utils.PreprocessText(record.FakeText)
			for _, token := range tokens {
				wordCounts[token]++
			}
		}

		// Processar texto verdadeiro
		if strings.TrimSpace(record.TrueText) != "" {
			tokens := utils.PreprocessText(record.TrueText)
			for _, token := range tokens {
				wordCounts[token]++
			}
		}
	}

	// Converter para slice para ordenação
	type wordCount struct {
		word  string
		count int
	}

	var wordCountList []wordCount
	for word, count := range wordCounts {
		wordCountList = append(wordCountList, wordCount{word, count})
	}

	// Ordenar por frequência (decrescente)
	sort.Slice(wordCountList, func(i, j int) bool {
		return wordCountList[i].count > wordCountList[j].count
	})

	// Pegar as 1000 palavras mais frequentes
	for i, wc := range wordCountList {
		if i >= c.InputSize {
			break
		}
		c.Vocab[wc.word] = i
	}
}

// textToVector converte texto para vetor de entrada
func (c *Classifier) textToVector(text string) []float64 {
	vector := make([]float64, c.InputSize)
	tokens := utils.PreprocessText(text)

	for _, token := range tokens {
		if index, exists := c.Vocab[token]; exists {
			vector[index] = 1.0
		}
	}

	return vector
}

// sigmoid função de ativação sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidDerivative derivada da função sigmoid
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// forwardPropagation realiza a propagação para frente
func (c *Classifier) forwardPropagation(input []float64) []float64 {
	currentInput := input

	for _, layer := range c.Layers {
		var layerOutput []float64

		for _, neuron := range layer.Neurons {
			// Calcular soma ponderada
			sum := neuron.Bias
			for i, weight := range neuron.Weights {
				sum += weight * currentInput[i]
			}

			// Aplicar função de ativação
			neuron.Output = sigmoid(sum)
			layerOutput = append(layerOutput, neuron.Output)
		}

		currentInput = layerOutput
	}

	return currentInput
}

// backwardPropagation realiza a retropropagação
func (c *Classifier) backwardPropagation(input []float64, target []float64) {
	// Forward pass
	_ = c.forwardPropagation(input)

	// Calcular erro da camada de saída
	for i, neuron := range c.Layers[len(c.Layers)-1].Neurons {
		neuron.Delta = (target[i] - neuron.Output) * sigmoidDerivative(neuron.Output)
	}

	// Backpropagate para camadas anteriores
	for layerIndex := len(c.Layers) - 2; layerIndex >= 0; layerIndex-- {
		currentLayer := c.Layers[layerIndex]
		nextLayer := c.Layers[layerIndex+1]

		for i, neuron := range currentLayer.Neurons {
			neuron.Delta = 0
			for _, nextNeuron := range nextLayer.Neurons {
				neuron.Delta += nextNeuron.Delta * nextNeuron.Weights[i]
			}
			neuron.Delta *= sigmoidDerivative(neuron.Output)
		}
	}

	// Atualizar pesos
	currentInput := input
	for _, layer := range c.Layers {
		for _, neuron := range layer.Neurons {
			// Atualizar bias
			neuron.Bias += c.LearningRate * neuron.Delta

			// Atualizar pesos
			for i := range neuron.Weights {
				neuron.Weights[i] += c.LearningRate * neuron.Delta * currentInput[i]
			}
		}

		// Preparar input para próxima camada
		var layerOutput []float64
		for _, neuron := range layer.Neurons {
			layerOutput = append(layerOutput, neuron.Output)
		}
		currentInput = layerOutput
	}
}

// Train treina o classificador
func (c *Classifier) Train(records []models.NewsRecord) {
	// Construir vocabulário
	c.buildVocabulary(records)

	// Preparar dados de treinamento
	var trainingData []struct {
		input  []float64
		target []float64
	}

	for _, record := range records {
		// Adicionar texto falso
		if strings.TrimSpace(record.FakeText) != "" {
			input := c.textToVector(record.FakeText)
			target := []float64{0.0, 1.0} // [verdadeira, falsa]
			trainingData = append(trainingData, struct {
				input  []float64
				target []float64
			}{input, target})
		}

		// Adicionar texto verdadeiro
		if strings.TrimSpace(record.TrueText) != "" {
			input := c.textToVector(record.TrueText)
			target := []float64{1.0, 0.0} // [verdadeira, falsa]
			trainingData = append(trainingData, struct {
				input  []float64
				target []float64
			}{input, target})
		}
	}

	// Treinamento
	for epoch := 0; epoch < c.Epochs; epoch++ {
		totalError := 0.0

		for _, data := range trainingData {
			// Forward pass
			outputs := c.forwardPropagation(data.input)

			// Calcular erro
			for i, output := range outputs {
				error := data.target[i] - output
				totalError += error * error
			}

			// Backward pass
			c.backwardPropagation(data.input, data.target)
		}

		// Imprimir progresso a cada 10 épocas
		if epoch%10 == 0 {
			fmt.Printf("Época %d/%d, Erro: %f\n", epoch, c.Epochs, totalError)
		}
	}

	fmt.Println("Treinamento concluído!")
}

// Classify classifica um texto
func (c *Classifier) Classify(text string) (string, float64) {
	input := c.textToVector(text)
	outputs := c.forwardPropagation(input)

	// Determinar classe
	if outputs[0] > outputs[1] {
		return "true", outputs[0] * 100
	} else {
		return "fake", outputs[1] * 100
	}
}

// ClassifyWithDebug classifica um texto com informações detalhadas
func (c *Classifier) ClassifyWithDebug(text string) (string, float64, map[string]float64, []string) {
	input := c.textToVector(text)
	outputs := c.forwardPropagation(input)

	// Calcular probabilidades
	probs := map[string]float64{
		"true": outputs[0] * 100,
		"fake": outputs[1] * 100,
	}

	// Determinar classe e confiança
	var label string
	var confidence float64
	if outputs[0] > outputs[1] {
		label = "true"
		confidence = outputs[0] * 100
	} else {
		label = "fake"
		confidence = outputs[1] * 100
	}

	// Encontrar tokens mais influentes
	tokens := utils.PreprocessText(text)
	var topTokens []string
	for _, token := range tokens {
		if index, exists := c.Vocab[token]; exists && index < len(input) {
			if input[index] > 0 {
				topTokens = append(topTokens, token)
			}
		}
	}

	// Limitar a 10 tokens
	if len(topTokens) > 10 {
		topTokens = topTokens[:10]
	}

	return label, confidence, probs, topTokens
}
