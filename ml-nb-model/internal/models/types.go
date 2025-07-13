package models

// NewsRecord representa um par de notícias (falsa e verdadeira)
type NewsRecord struct {
	TitleFake string
	FakeText  string
	LinkFake  string
	TrueText  string
	LinkTrue  string
}

// Metrics representa as métricas de avaliação
type Metrics struct {
	Accuracy  float64
	Precision float64
	Recall    float64
	F1Score   float64
}

// Fold representa um fold para cross-validation
type Fold struct {
	Train []NewsRecord
	Test  []NewsRecord
}

// ClassificationResult representa o resultado de uma classificação
type ClassificationResult struct {
	Label         string
	Confidence    float64
	Probabilities map[string]float64
	TopTokens     []string
}
