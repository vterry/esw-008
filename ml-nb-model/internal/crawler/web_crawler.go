package crawler

import (
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

// CrawlNews extrai o conteúdo de uma URL de notícia
func CrawlNews(url string) (string, error) {
	// Criar cliente HTTP com timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Fazer requisição
	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("falha na requisição: status code %d", resp.StatusCode)
	}

	// Carregar documento HTML
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return "", err
	}

	// Extrair texto dos elementos principais
	var textContent []string

	// Remover scripts e estilos
	doc.Find("script, style").Remove()

	// Extrair texto de elementos específicos de notícias
	selectors := []string{
		"article p",
		".content p",
		".post-content p",
		".article-content p",
		".story-body p",
		".entry-content p",
		"p",
	}

	for _, selector := range selectors {
		doc.Find(selector).Each(func(i int, s *goquery.Selection) {
			text := strings.TrimSpace(s.Text())
			if len(text) > 50 { // Filtrar textos muito pequenos
				textContent = append(textContent, text)
			}
		})
	}

	// Se não encontrou conteúdo específico, pegar todo o texto
	if len(textContent) == 0 {
		doc.Find("body").Each(func(i int, s *goquery.Selection) {
			text := strings.TrimSpace(s.Text())
			textContent = append(textContent, text)
		})
	}

	// Juntar todo o conteúdo
	fullText := strings.Join(textContent, " ")

	// Limpar texto
	fullText = strings.ReplaceAll(fullText, "\n", " ")
	fullText = strings.ReplaceAll(fullText, "\t", " ")

	// Remover espaços múltiplos
	reg := regexp.MustCompile(`\s+`)
	fullText = reg.ReplaceAllString(fullText, " ")

	fullText = strings.TrimSpace(fullText)

	return fullText, nil
}
