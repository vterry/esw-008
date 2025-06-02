package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/stat"
)

func main() {
	// 1. Abrir o arquivo CSV
	file, err := os.Open("../../datasets/vacinas-sp-jan-2025.csv")
	if err != nil {
		log.Fatalf("Erro ao abrir o arquivo CSV: %v", err)
	}
	defer file.Close()

	// 2. Criar um leitor de CSV
	reader := csv.NewReader(file)
	reader.Comma = ';' // Definir o delimitador como ponto e vírgula

	// 3. Ler o cabeçalho para encontrar os índices das colunas
	header, err := reader.Read()
	if err != nil {
		log.Fatalf("Erro ao ler o cabeçalho: %v", err)
	}

	colIndexes := make(map[string]int)
	// Colunas que queremos analisar
	colsToAnalyze := []string{
		"nu_idade_paciente",
		"co_dose_vacina",
		"co_vacina",
		"co_raca_cor_paciente",
		"co_local_aplicacao",
		"co_municipio_paciente",
		"co_cnes_estabelecimento",
	}

	for _, colName := range colsToAnalyze {
		found := false
		for i, name := range header {
			if name == colName {
				colIndexes[colName] = i
				found = true
				break
			}
		}
		if !found {
			log.Printf("Aviso: A coluna '%s' não foi encontrada no arquivo.", colName)
		}
	}

	// Verificar se a coluna principal 'nu_idade_paciente' foi encontrada
	if _, ok := colIndexes["nu_idade_paciente"]; !ok {
		log.Fatalf("Erro fatal: A coluna principal 'nu_idade_paciente' é necessária para a análise mas não foi encontrada.")
	}

	// 4. Ler os dados e armazenar em slices (listas) de float64
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Erro ao ler os registros do CSV: %v", err)
	}

	// Mapa para armazenar os dados de cada coluna
	data := make(map[string][]float64)
	for _, colName := range colsToAnalyze {
		data[colName] = []float64{}
	}

	// Iterar sobre cada linha do CSV
	for _, record := range records {
		// Tentar converter todos os valores das colunas de interesse para float
		tempRecord := make(map[string]float64)
		validRecord := true
		for colName, index := range colIndexes {
			val, err := strconv.ParseFloat(record[index], 64)
			if err != nil {
				// Se a conversão falhar para qualquer coluna, o registro é inválido para a análise
				validRecord = false
				break
			}
			tempRecord[colName] = val
		}

		// Se todos os valores foram convertidos com sucesso, adicioná-los aos nossos slices de dados
		if validRecord {
			for colName, val := range tempRecord {
				data[colName] = append(data[colName], val)
			}
		}
	}

	fmt.Printf("Análise baseada em %d registros válidos.\n\n", len(data["nu_idade_paciente"]))

	// 5. Calcular e exibir a correlação e covariância
	idadeData := data["nu_idade_paciente"]

	fmt.Println("--- Análise de Correlação e Covariância com 'nu_idade_paciente' ---")
	fmt.Printf("%-25s | %-15s | %-15s\n", "Variável", "Correlação", "Covariância")
	fmt.Println("-------------------------------------------------------------------")

	for colName, colData := range data {
		if colName == "nu_idade_paciente" {
			continue // Não calcular com ela mesma
		}

		// Garante que os slices tenham o mesmo tamanho
		if len(colData) != len(idadeData) {
			log.Printf("Aviso: A quantidade de dados para '%s' (%d) é diferente da idade (%d). Pulando esta variável.", colName, len(colData), len(idadeData))
			continue
		}

		// Usar a biblioteca gonum/stat para os cálculos
		correlation := stat.Correlation(idadeData, colData, nil)
		covariance := stat.Covariance(idadeData, colData, nil)

		fmt.Printf("%-25s | %-15.4f | %-15.4f\n", colName, correlation, covariance)
	}
}
