package feature_extraction

import (
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
	vocabulary  map[string]int
}

// rawDocuments := {
//      'This is the first document.',
//      'This document is the second document.',
//      'And this is the third one.',
//      'Is this the first document?',
//  }

// [[0 1 1 1 0 0 1 0 1]
//  [0 2 0 1 0 1 1 0 1]
//  [1 0 0 1 1 0 1 1 1]
//  [0 1 1 1 0 0 1 0 1]]

// define vocabulary and sparse matrix based on analyzer and etc,here just the regular
// matrix will be used
// maxFeatures обрубает по большому количеству документов
func (cv *CountVectorizer) CountVocab(rawDocs []string) map[string]int {
	vocab := make(map[string]int)
	for _, doc := range rawDocs {
		analyzed := cv.AnalyzeWord(doc)
		for _, tok := range analyzed {
			if _, ok := vocab[tok]; !ok {
				vocab[tok] = 1
			} else {
				vocab[tok] += 1
			}
		}
	}
	return vocab
}

func (cv *CountVectorizer) LimitFeatures(vocab map[string]int) map[string]int {
	values := make([]int, len(vocab))
	for _, v := range vocab {
		values = append(values, v)
	}

	if len(vocab) <= cv.maxFeatures {
		cv.maxFeatures = len(vocab)
	}

	sort.Sort(sort.Reverse(sort.IntSlice(values)))

	minValue := values[cv.maxFeatures-1]

	newVocab := make(map[string]int, cv.maxFeatures)

	for k, v := range vocab {
		if v < minValue {
			continue
		}
		if v < cv.minDf {
			continue
		}
		if v > cv.maxDf {
			continue
		}
		newVocab[k] = v
	}

	return newVocab
}

// useless function imho
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

func (cv *CountVectorizer) Analyze(text string) []string {
	ngrams := make([]string, 0)
	runeQuery := []rune(text)
	for i := 0; i < len(runeQuery); i++ {
		for j := cv.nGramRange.MinN; j <= cv.nGramRange.MaxN && i+j <= len(runeQuery); j++ {
			ngrams = append(ngrams, string(runeQuery[i:i+j]))
		}
	}
	return ngrams
}

func (cv *CountVectorizer) AnalyzeWord(text string) []string {
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
	return ngrams
}

func (cv *CountVectorizer) CalcMat(docs []string) []float32 {
	dim := cv.maxFeatures
	x := make([]float32, len(docs)*dim)
	a := make([]string, dim)
	vec := make([]float32, dim)
	for i, d := range docs {
		a = cv.AnalyzeWord(d)
		vec = cv.GetVector(a)
		for j, v := range vec {
			x[dim*i+j] = v
		}
	}
	return x
}

func (cv *CountVectorizer) FitTransform(documents []string) []float32 {
	vocabulary := cv.CountVocab(documents)
	vocabulary = cv.LimitFeatures(vocabulary)
	cv.vocabulary = cv.SortFeatures(vocabulary)
	x := cv.CalcMat(documents)
	return x
}
