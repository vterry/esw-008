// Salve este arquivo como main.go
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// VaccinationRecord armazena os dados relevantes de uma linha do CSV.
type VaccinationRecord struct {
	Age   float64
	Dose  string
	Group string
}

// StatsResult armazena os resultados estatísticos para um grupo.
type StatsResult struct {
	Count  int
	Mean   float64
	StdDev float64
	Min    float64
	Q25    float64
	Median float64
	Q75    float64
	Max    float64
}

// main é a função principal que executa a análise.
func main() {
	records := loadData("../../../datasets/vacinas-sp-jan-2025.csv")
	if len(records) == 0 {
		log.Fatal("Nenhum registro válido foi carregado.")
	}

	fmt.Println("--- Análise Estatística dos Dados de Vacinação ---")

	// Análise: Idade vs. Dose
	analyzeGroup(records, "Dose da Vacina", func(r VaccinationRecord) string { return r.Dose })

	// Análise: Idade vs. Grupo de Atendimento
	analyzeGroup(records, "Grupo de Atendimento", func(r VaccinationRecord) string { return r.Group })

	// Matriz de Correlação
	calculateCorrelationMatrix(records)
}

// analyzeGroup agrupa os dados e calcula estatísticas descritivas.
func analyzeGroup(records []VaccinationRecord, groupTitle string, keyFunc func(VaccinationRecord) string) {
	fmt.Printf("\n\n--- Análise da Idade por %s ---\n\n", groupTitle)
	fmt.Printf("%-45s | %-8s | %-8s | %-8s | %-8s\n", groupTitle, "Count", "Mean", "Median", "Std Dev")
	fmt.Println(string(make([]byte, 90, 90)))

	agesByGroup := make(map[string][]float64)
	for _, r := range records {
		key := keyFunc(r)
		agesByGroup[key] = append(agesByGroup[key], r.Age)
	}

	groupNames := make([]string, 0, len(agesByGroup))
	for name := range agesByGroup {
		groupNames = append(groupNames, name)
	}
	sort.Strings(groupNames)

	for _, name := range groupNames {
		stats := calculateStats(agesByGroup[name])
		if stats.Count > 0 {
			fmt.Printf("%-45s | %-8d | %-8.2f | %-8.2f | %-8.2f\n", name, stats.Count, stats.Mean, stats.Median, stats.StdDev)
		}
	}
}

// calculateCorrelationMatrix calcula e exibe a matriz de correlação.
func calculateCorrelationMatrix(records []VaccinationRecord) {
	fmt.Println("\n\n--- Matriz de Correlação ---")

	ages, doseCodes, groupCodes := factorize(records)

	data := mat.NewDense(len(records), 3, nil)
	data.SetCol(0, ages)
	data.SetCol(1, doseCodes)
	data.SetCol(2, groupCodes)

	corrMatrix := mat.NewSymDense(3, nil)
	stat.CorrelationMatrix(corrMatrix, data, nil)

	fmt.Println("\nValores de Correlação de Pearson:")
	fmt.Printf("Correlação Idade vs Dose:  %+.4f\n", corrMatrix.At(0, 1))
	fmt.Printf("Correlação Idade vs Grupo: %+.4f\n", corrMatrix.At(0, 2))
	fmt.Printf("Correlação Dose vs Grupo:  %+.4f\n", corrMatrix.At(1, 2))
}

// --- Funções Auxiliares ---

func loadData(filePath string) []VaccinationRecord {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Erro ao abrir o arquivo %s: %v", filePath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'
	header, err := reader.Read()
	if err != nil {
		log.Fatal("Erro ao ler o cabeçalho.")
	}

	colIdx := make(map[string]int)
	targetCols := []string{"nu_idade_paciente", "ds_dose_vacina", "ds_vacina_grupo_atendimento"}
	for _, colName := range targetCols {
		found := false
		for i, h := range header {
			if h == colName {
				colIdx[colName] = i
				found = true
				break
			}
		}
		if !found {
			log.Fatalf("Coluna obrigatória '%s' não encontrada.", colName)
		}
	}

	csvRecords, err := reader.ReadAll()
	if err != nil {
		log.Fatal("Erro ao ler os dados do CSV.")
	}

	var records []VaccinationRecord
	for _, row := range csvRecords {
		ageStr, dose, group := row[colIdx["nu_idade_paciente"]], row[colIdx["ds_dose_vacina"]], row[colIdx["ds_vacina_grupo_atendimento"]]
		if ageStr == "" || dose == "" || group == "" {
			continue
		}
		age, err := strconv.ParseFloat(ageStr, 64)
		if err != nil {
			continue
		}
		records = append(records, VaccinationRecord{Age: age, Dose: dose, Group: group})
	}
	return records
}

func calculateStats(data []float64) StatsResult {
	if len(data) == 0 {
		return StatsResult{}
	}
	sort.Float64s(data)
	mean, std := stat.MeanStdDev(data, nil)
	return StatsResult{
		Count:  len(data),
		Mean:   mean,
		StdDev: std,
		Min:    data[0],
		Q25:    stat.Quantile(0.25, stat.Empirical, data, nil),
		Median: stat.Quantile(0.5, stat.Empirical, data, nil),
		Q75:    stat.Quantile(0.75, stat.Empirical, data, nil),
		Max:    data[len(data)-1],
	}
}

func factorize(records []VaccinationRecord) (ages, doseCodes, groupCodes []float64) {
	doseMap, groupMap := make(map[string]float64), make(map[string]float64)
	for _, r := range records {
		ages = append(ages, r.Age)
		if _, ok := doseMap[r.Dose]; !ok {
			doseMap[r.Dose] = float64(len(doseMap))
		}
		doseCodes = append(doseCodes, doseMap[r.Dose])
		if _, ok := groupMap[r.Group]; !ok {
			groupMap[r.Group] = float64(len(groupMap))
		}
		groupCodes = append(groupCodes, groupMap[r.Group])
	}
	return
}
