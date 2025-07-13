package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/souza/esw-008/ml-nb-model/internal/crawler"
	"github.com/souza/esw-008/ml-nb-model/internal/mlp"
	"github.com/souza/esw-008/ml-nb-model/internal/models"
	"github.com/souza/esw-008/ml-nb-model/internal/naivebayes"
)

// loadDatasetFromURLTest carrega o dataset de uma URL
func loadDatasetFromURLTest(url string) ([]models.NewsRecord, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return parseDatasetTest(resp.Body)
}

// parseDatasetTest parseia o dataset CSV
func parseDatasetTest(r io.Reader) ([]models.NewsRecord, error) {
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

// analyzeURLTest analisa uma URL específica
func analyzeURLTest(url string, records []models.NewsRecord) {
	fmt.Printf("\n" + strings.Repeat("=", 80))
	fmt.Printf("ANALISANDO: %s\n", url)
	fmt.Println(strings.Repeat("=", 80))

	// Extrair conteúdo
	articleText, err := crawler.CrawlNews(url)
	if err != nil {
		fmt.Printf("❌ Erro ao extrair conteúdo: %v\n", err)
		return
	}

	if strings.TrimSpace(articleText) == "" {
		fmt.Println("❌ Não foi possível extrair texto relevante da página.")
		return
	}

	if len(articleText) < 300 {
		fmt.Printf("⚠️  [AVISO] O texto extraído é muito pequeno (%d caracteres).\n", len(articleText))
	}

	// Verificar heurística
	lowerText := strings.ToLower(articleText)
	if strings.Contains(lowerText, "boato") || strings.Contains(lowerText, "falso") ||
		strings.Contains(lowerText, "mentira") || strings.Contains(lowerText, "desmentido") ||
		strings.Contains(lowerText, "fake news") {
		fmt.Println("🔍 [HEURÍSTICA] O texto contém termos típicos de desmentido ou fake news.")
		fmt.Println("📋 Classificação: Provavelmente Falsa (por heurística)")
		return
	}

	// Treinar classificadores
	fmt.Println("🤖 Treinando classificadores...")

	mlpClassifier := mlp.NewClassifier(1000, 50, 2)
	mlpClassifier.Train(records)

	nbClassifier := naivebayes.NewClassifier()
	nbClassifier.TrainNB(records)

	// Classificar
	mlpLabel, mlpConfidence, mlpProbs, mlpTokens := mlpClassifier.ClassifyWithDebug(articleText)
	nbLabel, nbConfidence, nbProbs, nbTokens := nbClassifier.ClassifyWithDebugNB(articleText)

	// Mapear resultados
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

	// Exibir resultados
	fmt.Printf("\n📊 RESULTADOS:\n")
	fmt.Printf("%-20s %-25s %-15s %-20s\n", "Algoritmo", "Classificação", "Confiança", "Probabilidades")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s\n", "MLP", mlpResult, mlpConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", mlpProbs["true"], mlpProbs["fake"]))
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s\n", "Naive Bayes", nbResult, nbConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", nbProbs["true"], nbProbs["fake"]))

	// Análise de concordância
	fmt.Printf("\n🔍 ANÁLISE DE CONCORDÂNCIA:\n")
	if mlpLabel == nbLabel {
		fmt.Printf("✅ Os algoritmos concordam: %s\n", mlpResult)
	} else {
		fmt.Printf("❌ Os algoritmos discordam:\n")
		fmt.Printf("   MLP: %s (%.2f%%)\n", mlpResult, mlpConfidence)
		fmt.Printf("   NB:  %s (%.2f%%)\n", nbResult, nbConfidence)
	}

	// Tokens mais influentes
	fmt.Printf("\n🔤 TOKENS MAIS INFLUENTES:\n")
	fmt.Println("MLP:")
	for i, token := range mlpTokens {
		if i < 3 {
			fmt.Printf("  • %s\n", token)
		}
	}
	fmt.Println("Naive Bayes:")
	for i, token := range nbTokens {
		if i < 3 {
			fmt.Printf("  • %s\n", token)
		}
	}

	fmt.Println(strings.Repeat("=", 80))
}

// ResultadoURL representa o resultado da análise de uma URL
type ResultadoURL struct {
	Noticia      string
	ResultadoMLP string
	ResultadoNB  string
}

// analyzeURLTestWithReturn analisa uma URL específica e retorna o resultado
func analyzeURLTestWithReturn(url string, records []models.NewsRecord, nomeNoticia string) ResultadoURL {
	fmt.Printf("\n" + strings.Repeat("=", 80))
	fmt.Printf("ANALISANDO: %s\n", url)
	fmt.Println(strings.Repeat("=", 80))

	// Extrair conteúdo
	articleText, err := crawler.CrawlNews(url)
	if err != nil {
		fmt.Printf("❌ Erro ao extrair conteúdo: %v\n", err)
		return ResultadoURL{
			Noticia:      nomeNoticia,
			ResultadoMLP: "Erro na extração",
			ResultadoNB:  "Erro na extração",
		}
	}

	if strings.TrimSpace(articleText) == "" {
		fmt.Println("❌ Não foi possível extrair texto relevante da página.")
		return ResultadoURL{
			Noticia:      nomeNoticia,
			ResultadoMLP: "Texto não extraído",
			ResultadoNB:  "Texto não extraído",
		}
	}

	if len(articleText) < 300 {
		fmt.Printf("⚠️  [AVISO] O texto extraído é muito pequeno (%d caracteres).\n", len(articleText))
	}

	// Verificar heurística
	lowerText := strings.ToLower(articleText)
	if strings.Contains(lowerText, "boato") || strings.Contains(lowerText, "falso") ||
		strings.Contains(lowerText, "mentira") || strings.Contains(lowerText, "desmentido") ||
		strings.Contains(lowerText, "fake news") {
		fmt.Println("🔍 [HEURÍSTICA] O texto contém termos típicos de desmentido ou fake news.")
		fmt.Println("📋 Classificação: Provavelmente Falsa (por heurística)")
		return ResultadoURL{
			Noticia:      nomeNoticia,
			ResultadoMLP: "Provavelmente Falsa (heurística)",
			ResultadoNB:  "Provavelmente Falsa (heurística)",
		}
	}

	// Treinar classificadores
	fmt.Println("🤖 Treinando classificadores...")

	mlpClassifier := mlp.NewClassifier(1000, 50, 2)
	mlpClassifier.Train(records)

	nbClassifier := naivebayes.NewClassifier()
	nbClassifier.TrainNB(records)

	// Classificar
	mlpLabel, mlpConfidence, mlpProbs, mlpTokens := mlpClassifier.ClassifyWithDebug(articleText)
	nbLabel, nbConfidence, nbProbs, nbTokens := nbClassifier.ClassifyWithDebugNB(articleText)

	// Mapear resultados
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

	// Exibir resultados
	fmt.Printf("\n📊 RESULTADOS:\n")
	fmt.Printf("%-20s %-25s %-15s %-20s\n", "Algoritmo", "Classificação", "Confiança", "Probabilidades")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s\n", "MLP", mlpResult, mlpConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", mlpProbs["true"], mlpProbs["fake"]))
	fmt.Printf("%-20s %-25s %-15.2f%% %-20s\n", "Naive Bayes", nbResult, nbConfidence,
		fmt.Sprintf("V:%.1f%% F:%.1f%%", nbProbs["true"], nbProbs["fake"]))

	// Análise de concordância
	fmt.Printf("\n🔍 ANÁLISE DE CONCORDÂNCIA:\n")
	if mlpLabel == nbLabel {
		fmt.Printf("✅ Os algoritmos concordam: %s\n", mlpResult)
	} else {
		fmt.Printf("❌ Os algoritmos discordam:\n")
		fmt.Printf("   MLP: %s (%.2f%%)\n", mlpResult, mlpConfidence)
		fmt.Printf("   NB:  %s (%.2f%%)\n", nbResult, nbConfidence)
	}

	// Tokens mais influentes
	fmt.Printf("\n🔤 TOKENS MAIS INFLUENTES:\n")
	fmt.Println("MLP:")
	for i, token := range mlpTokens {
		if i < 3 {
			fmt.Printf("  • %s\n", token)
		}
	}
	fmt.Println("Naive Bayes:")
	for i, token := range nbTokens {
		if i < 3 {
			fmt.Printf("  • %s\n", token)
		}
	}

	fmt.Println(strings.Repeat("=", 80))

	return ResultadoURL{
		Noticia:      nomeNoticia,
		ResultadoMLP: mlpResult,
		ResultadoNB:  nbResult,
	}
}

// printSummaryTable imprime a tabela de resumo dos resultados
func printSummaryTable(resultados []ResultadoURL) {
	fmt.Println("\n" + strings.Repeat("=", 120))
	fmt.Println("📋 RESUMO DOS RESULTADOS")
	fmt.Println(strings.Repeat("=", 120))

	// Cabeçalho da tabela
	fmt.Printf("%-60s %-30s %-30s\n", "Notícia", "Resultado MLP", "Resultado NB")
	fmt.Println(strings.Repeat("-", 120))

	// Dados da tabela
	for _, resultado := range resultados {
		// Truncar nomes muito longos
		noticia := resultado.Noticia
		if len(noticia) > 58 {
			noticia = noticia[:55] + "..."
		}

		mlpResult := resultado.ResultadoMLP
		if len(mlpResult) > 28 {
			mlpResult = mlpResult[:25] + "..."
		}

		nbResult := resultado.ResultadoNB
		if len(nbResult) > 28 {
			nbResult = nbResult[:25] + "..."
		}

		fmt.Printf("%-60s %-30s %-30s\n",
			noticia,
			mlpResult,
			nbResult)
	}

	fmt.Println(strings.Repeat("=", 120))

	// Estatísticas
	concordancia := 0
	for _, resultado := range resultados {
		if resultado.ResultadoMLP == resultado.ResultadoNB {
			concordancia++
		}
	}

	fmt.Printf("\n📊 ESTATÍSTICAS:\n")
	fmt.Printf("• Total de notícias analisadas: %d\n", len(resultados))
	fmt.Printf("• Concordância entre algoritmos: %d/%d (%.1f%%)\n",
		concordancia, len(resultados), float64(concordancia)/float64(len(resultados))*100)

	// Contar classificações por algoritmo
	mlpVerdadeiras, mlpFalsas := 0, 0
	nbVerdadeiras, nbFalsas := 0, 0

	for _, resultado := range resultados {
		if strings.Contains(resultado.ResultadoMLP, "Verdadeira") {
			mlpVerdadeiras++
		} else {
			mlpFalsas++
		}

		if strings.Contains(resultado.ResultadoNB, "Verdadeira") {
			nbVerdadeiras++
		} else {
			nbFalsas++
		}
	}

	fmt.Printf("• MLP: %d verdadeiras, %d falsas\n", mlpVerdadeiras, mlpFalsas)
	fmt.Printf("• Naive Bayes: %d verdadeiras, %d falsas\n", nbVerdadeiras, nbFalsas)
}

func main() {
	// URLs para teste
	urls := []string{
		"https://g1.globo.com/saude/bem-estar/noticia/2025/07/07/como-fazer-a-higiene-do-sono-veja-quais-sao-os-maiores-inimigos-de-uma-noite-restauradora.ghtml",
		"https://g1.globo.com/mg/triangulo-mineiro/noticia/2022/11/03/mpmg-denuncia-quatro-por-ataque-com-drone-a-apoiadores-de-lula-em-mg.ghtml",
		"https://www.boatos.org/politica/haddad-criou-imposto-para-taxar-o-oxigenio-respirado-pelos-brasileiros.html",
		"https://www.estadao.com.br/estadao-verifica/contas-dark-perfis-anonimos-de-saude-com-42-milhoes-de-seguidores-aplicam-golpes-no-instagram/",
		"https://www.boatos.org/brasil/eua-enviaram-porta-avioes-ao-lago-paranoa-por-causa-de-trump.html",
	}

	// Nomes das notícias para a tabela
	nomesNoticias := []string{
		"G1 - Higiene do sono",
		"G1 - Denúncia MPMG sobre drone",
		"Boatos.org - Haddad e imposto do oxigênio",
		"Estadão - Contas dark no Instagram",
		"Boatos.org - Porta-aviões no Lago Paranoá",
	}

	fmt.Println("🚀 INICIANDO TESTE DE CLASSIFICAÇÃO DE FAKE NEWS")
	fmt.Println("📋 Analisando 5 URLs diferentes...")
	fmt.Println("⏱️  Tempo estimado: 5-10 minutos")

	// Carregar dataset
	fmt.Println("\n📥 Carregando dataset...")
	datasetURL := "https://raw.githubusercontent.com/jpchav98/FakeTrue.Br/refs/heads/main/FakeTrueBr_corpus.csv"
	records, err := loadDatasetFromURLTest(datasetURL)
	if err != nil {
		log.Fatalf("❌ Falha ao carregar o dataset: %v", err)
	}
	fmt.Printf("✅ Dataset carregado com %d registros\n", len(records))

	// Slice para armazenar resultados
	var resultados []ResultadoURL

	// Analisar cada URL
	for i, url := range urls {
		fmt.Printf("\n🔄 Processando URL %d/5...\n", i+1)
		resultado := analyzeURLTestWithReturn(url, records, nomesNoticias[i])
		resultados = append(resultados, resultado)

		// Pausa entre análises para não sobrecarregar os servidores
		if i < len(urls)-1 {
			fmt.Println("⏳ Aguardando 3 segundos antes da próxima análise...")
			time.Sleep(3 * time.Second)
		}
	}

	// Imprimir tabela de resumo
	printSummaryTable(resultados)

	fmt.Println("\n🎉 TESTE CONCLUÍDO!")
	fmt.Println("📊 Resumo das análises:")
	fmt.Println("   • URL 1: Notícia sobre saúde (G1)")
	fmt.Println("   • URL 2: Notícia política (G1)")
	fmt.Println("   • URL 3: Boato político (Boatos.org)")
	fmt.Println("   • URL 4: Verificação de fatos (Estadão)")
	fmt.Println("   • URL 5: Boato político (Boatos.org)")
}
