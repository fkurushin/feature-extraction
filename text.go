package feature_extraction

import (
	"sort"
)

type Range struct {
	MinN int
	MaxN int
}

type CountVectorizer struct {
	maxFeatures int
	nGramRange  Range
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
func (cv *CountVectorizer) countVocab(rawDocuments []string) {
	vocab := make(map[string]int)
	for _, doc := range rawDocuments {
		analyzed := cv.Analyze(doc)
		for _, tok := range analyzed {
			if _, ok := vocab[tok]; !ok {
				vocab[tok] = 1
			} else {
				vocab[tok] += 1
			}
		}
	}
	// retrun nil
}

func (cv *CountVectorizer) LimitFeatures(vocab map[string]int) map[string]int {
	freqs := make([]int, 0, len(vocab))
	for _, v := range vocab {
		freqs = append(freqs, v)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(freqs)))
	minIdx := cv.maxFeatures
	if minIdx >= len(freqs) {
		minIdx = len(freqs) - 1
	}
	min := freqs[minIdx]
	vocabLim := make(map[string]int, cv.maxFeatures)
	extra := make([]string, 0, len(vocab)-cv.maxFeatures)
	for k, v := range vocab {
		if v > min && len(vocabLim) < cv.maxFeatures {
			vocabLim[k] = v
		} else if v == min {
			extra = append(extra, k)
		}
	}
	for i := range extra {
		if len(vocabLim) == cv.maxFeatures {
			break
		}
		vocabLim[extra[i]] = min
	}
	return vocabLim
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
