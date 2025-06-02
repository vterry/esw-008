package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"time" 
)

import (
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	// Flags de dados e filtro
	colIndex := flag.Int("col", 1, "Índice da coluna para gerar o box plot (base 0).")
	filtroCol := flag.Int("filtro-col", -1, "Índice da coluna para USAR COMO FILTRO. Desativado por padrão (-1).")
	filtroValor := flag.String("filtro-valor", "", "Valor a ser encontrado na coluna de filtro.")
	
	// Flags gerais
	csvPath := flag.String("csv", "vendas_por_produto.csv", "Caminho para o arquivo CSV de entrada.")
	outputPath := flag.String("out", "boxplot.png", "Caminho para salvar o arquivo de imagem do gráfico.")
	flag.Parse()

	// Leitura do CSV
	file, err := os.Open(*csvPath)
	if err != nil {
		log.Fatalf("Erro ao abrir o arquivo CSV %s: %v", *csvPath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'

	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Erro ao ler os registros do CSV: %v", err)
	}

	// Extração e Conversão
	var data plotter.Values
	const dateLayout = "02/01/2006"

	for i, record := range records {
		if i == 0 {
			continue // Pula cabeçalho
		}

		// Lógica de Filtragem
		if *filtroCol >= 0 {
			if *filtroCol >= len(record) {
				continue 
			}
			if record[*filtroCol] != *filtroValor {
				continue 
			}
		}

		// Processamento da coluna de dados
		if *colIndex >= len(record) {
			log.Printf("Aviso: a linha %d não tem a coluna de dados %d. Pulando...", i+1, *colIndex)
			continue
		}
		
		dateString := record[*colIndex]
		t, err := time.Parse(dateLayout, dateString)
		if err != nil {
			log.Printf("Aviso: não foi possível converter '%s' na linha %d para data. Pulando...", dateString, i+1)
			continue
		}
		data = append(data, float64(t.Unix()))
	}

	if len(data) == 0 {
		log.Fatal("Nenhum dado encontrado para os critérios especificados. Abortando.")
	}

	// Criação do Gráfico
	p := plot.New()
	p.Title.Text = "Box Plot"
	p.Y.Tick.Marker = plot.TimeTicks{Format: "02/01/2006"}
	box, _ := plotter.NewBoxPlot(vg.Points(50), 0, data)
	p.Add(box)
	
	if err := p.Save(4*vg.Inch, 6*vg.Inch, *outputPath); err != nil {
		log.Fatalf("Erro ao salvar o gráfico: %v", err)
	}

	fmt.Printf("Gráfico salvo com sucesso em '%s'\n", *outputPath)
}