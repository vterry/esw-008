package utils

import (
	"regexp"
	"strings"
)

// GetStopWords retorna o mapa de stop words
func GetStopWords() map[string]bool {
	return stopWords
}

// StopWords em português
var stopWords = map[string]bool{
	"a": true, "o": true, "e": true, "é": true, "de": true, "do": true, "da": true, "em": true, "um": true, "para": true,
	"com": true, "não": true, "uma": true, "os": true, "no": true, "se": true, "na": true, "por": true, "mais": true,
	"as": true, "dos": true, "como": true, "mas": true, "foi": true, "ao": true, "ele": true, "das": true, "tem": true, "à": true,
	"seu": true, "sua": true, "ou": true, "ser": true, "quando": true, "muito": true, "há": true, "nos": true, "já": true, "está": true,
	"eu": true, "também": true, "só": true, "pelo": true, "pela": true, "até": true, "isso": true, "ela": true, "entre": true, "era": true,
	"depois": true, "sem": true, "mesmo": true, "aos": true, "ter": true, "seus": true, "suas": true, "minha": true, "têm": true, "numa": true,
	"num": true, "eles": true, "você": true, "esse": true, "esses": true, "essas": true, "meu": true, "minhas": true,
	"teu": true, "tuas": true, "nosso": true, "nossa": true, "nossos": true, "nossas": true, "dela": true, "delas": true, "esta": true, "estes": true,
	"estas": true, "aquele": true, "aquela": true, "aqueles": true, "aquelas": true, "isto": true, "aquilo": true, "estou": true,
	"estamos": true, "estão": true, "estava": true, "estávamos": true, "estavam": true, "estive": true, "esteve": true, "estivemos": true,
	"estiveram": true, "estivera": true, "estivéramos": true, "esteja": true, "estejamos": true,
	"estejam": true, "estivesse": true, "estivéssemos": true, "estivessem": true, "estiver": true, "estivermos": true, "estiverem": true,
	"hei": true, "havemos": true, "hão": true, "houve": true, "houvemos": true, "houveram": true, "houvera": true, "houvéramos": true,
	"haja": true, "hajamos": true, "hajam": true, "houvesse": true, "houvéssemos": true, "houvessem": true, "houver": true, "houvermos": true,
	"houverem": true, "houverei": true, "houverá": true, "houveremos": true, "houverão": true, "houveria": true, "houveríamos": true,
	"houveriam": true, "sou": true, "somos": true, "são": true, "éramos": true, "eram": true, "fui": true, "fomos": true,
	"foram": true, "fora": true, "fôramos": true, "seja": true, "sejamos": true, "sejam": true, "fosse": true, "fôssemos": true, "fossem": true,
	"for": true, "formos": true, "forem": true, "serei": true, "será": true, "seremos": true, "serão": true, "seria": true, "seríamos": true,
	"seriam": true, "tenho": true, "temos": true, "tinha": true, "tínhamos": true, "tinham": true, "tive": true,
	"teve": true, "tivemos": true, "tiveram": true, "tivera": true, "tivéramos": true, "tenha": true, "tenhamos": true, "tenham": true,
	"tivesse": true, "tivéssemos": true, "tivessem": true, "tiver": true, "tivermos": true, "tiverem": true, "terei": true, "terá": true,
	"teremos": true, "terão": true, "teria": true, "teríamos": true, "teriam": true,
}

// PreprocessText processa o texto removendo stop words e normalizando
func PreprocessText(text string) []string {
	// Converter para minúsculas
	text = strings.ToLower(text)

	// Remover pontuação e caracteres especiais
	reg := regexp.MustCompile(`[^\p{L}\s]`)
	text = reg.ReplaceAllString(text, " ")

	// Dividir em tokens
	tokens := strings.Fields(text)

	// Filtrar stop words
	var filteredTokens []string
	for _, token := range tokens {
		if len(token) > 1 && !stopWords[token] {
			filteredTokens = append(filteredTokens, token)
		}
	}

	return filteredTokens
}

// Min retorna o menor de dois inteiros
func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
