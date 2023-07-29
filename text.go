package feature_extraction

import (
	"fmt"
	"sort"
	"strings"
)

type Range struct {
	MinN int
	MaxN int
}

type CountVectorizer struct {
	maxFeatures int
	nGramRange  Range
	maxDf       int
	minDf       int
	norm        bool
	analyzer    string
	vocabulary  map[string]int
}

func (cv *CountVectorizer) CountVocab(rawDocs []string) (map[string]int, map[string]int, error) {
	vocab := make(map[string]int)
	docCounts := make(map[string]int)

	for _, doc := range rawDocs {
		analyzed, err := cv.Analyze(doc)
		if err != nil {
			return nil, nil, err
		}

		countedTokens := make(map[string]bool)
		for _, tok := range analyzed {
			if _, ok := vocab[tok]; !ok {
				vocab[tok] = 1
			} else {
				vocab[tok] += 1
			}

			if _, counted := countedTokens[tok]; !counted {
				docCounts[tok] += 1
				countedTokens[tok] = true
			}
		}
	}

	return vocab, docCounts, nil
}

func (cv *CountVectorizer) LimitFeatures(vocab map[string]int, docFreq map[string]int) map[string]int {

	if len(vocab) <= cv.maxFeatures {
		cv.maxFeatures = len(vocab)
	}

	reducedDfsVoc := make(map[string]int, cv.maxFeatures)

	for k, v := range docFreq {
		if v < cv.minDf || v > cv.maxDf {
			continue
		}
		reducedDfsVoc[k] = vocab[k]
	}
	if len(reducedDfsVoc) <= cv.maxFeatures {
		return reducedDfsVoc
	} else {
		values := make([]int, len(reducedDfsVoc))
		for _, v := range vocab {
			values = append(values, v)
		}
		sort.Sort(sort.Reverse(sort.IntSlice(values)))
		minValue := values[cv.maxFeatures]
		newVocab := make(map[string]int, cv.maxFeatures)

		for k, v := range reducedDfsVoc {
			//
			if v >= minValue && len(newVocab) < cv.maxFeatures {
				newVocab[k] = v
			}
		}
		return newVocab
	}
}

// i would name this as reorganize features
func (cv *CountVectorizer) SortFeatures(vocab map[string]int) map[string]int {
	keys := make([]string, 0, len(vocab))
	for k := range vocab {
		keys = append(keys, k)
	}

	sort.Strings(keys)

	newVocab := make(map[string]int, len(vocab))

	for i, v := range keys {
		newVocab[v] = i
	}
	return newVocab
}

func (cv *CountVectorizer) GetVector(analyzed []string) []float32 {
	vector := make([]float32, len(cv.vocabulary))
	for _, a := range analyzed {
		if ind, ok := cv.vocabulary[a]; ok {
			vector[ind] += 1
		}
	}
	if cv.norm {
		return normalize(vector)
	} else {
		return vector
	}
}

func (cv *CountVectorizer) Analyze(text string) ([]string, error) {
	switch cv.analyzer {
	case "char":
		// TODO: delete spaces
		ngrams := make([]string, 0)
		runeQuery := []rune(text)
		for i := 0; i < len(runeQuery); i++ {
			for j := cv.nGramRange.MinN; j <= cv.nGramRange.MaxN && i+j <= len(runeQuery); j++ {
				ngrams = append(ngrams, string(runeQuery[i:i+j]))
			}
		}
		return ngrams, nil

	case "word":
		n := cv.nGramRange.MaxN
		words := strings.Split(text, " ")
		ngrams := make([]string, 0, len(words)*((n-cv.nGramRange.MinN)+1))

		var builder strings.Builder
		for i := 0; i < len(words); i++ {
			for j := cv.nGramRange.MinN; j <= cv.nGramRange.MaxN && i+j <= len(words); j++ {
				builder.Reset()
				for k := 0; k < j; k++ {
					if k > 0 {
						builder.WriteByte(' ')
					}
					builder.WriteString(words[i+k])
				}
				ngrams = append(ngrams, builder.String())
			}
		}
		return ngrams, nil

	default:
		return nil, fmt.Errorf("unexpected analyzer name")
	}
}

func (cv *CountVectorizer) CalcMat(docs []string) ([]float32, error) {
	// необязательно заново анализировать, можно просто удалить стоп-токенс
	// а затем пересортировать
	// но это потом
	var err error
	dim := cv.maxFeatures
	x := make([]float32, len(docs)*dim)
	a := make([]string, dim)
	vec := make([]float32, dim)
	for i, d := range docs {
		a, err = cv.Analyze(d)
		if err != nil {
			return nil, err
		}
		vec = cv.GetVector(a)
		for j, v := range vec {
			x[dim*i+j] = v
		}
	}
	return x, nil
}

func (cv *CountVectorizer) FitTransform(documents []string) ([]float32, error) {

	vocabulary, dfs, err := cv.CountVocab(documents)
	if err != nil {
		return nil, err
	}
	vocabulary = cv.LimitFeatures(vocabulary, dfs)
	cv.vocabulary = cv.SortFeatures(vocabulary)
	x, err := cv.CalcMat(documents)
	if err != nil {
		return nil, err
	}
	fmt.Println(cv.vocabulary)
	return x, nil
}
