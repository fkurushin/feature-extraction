package feature_extraction

import (
	"fmt"
	"reflect"
	"testing"
)

func TestKeepFirstNelements(t *testing.T) {
	tests := []struct {
		inputMap map[string]int
		inputN   int
		expected map[string]int
	}{
		{
			inputMap: map[string]int{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
			inputN:   3,
			expected: map[string]int{"d": 4, "e": 5, "c": 3},
		},
		{
			inputMap: map[string]int{"a": 10, "b": 20, "c": 30, "d": 40, "e": 50},
			inputN:   5,
			expected: map[string]int{"e": 50, "d": 40, "c": 30, "b": 20, "a": 10},
		},
	}
	for _, test := range tests {
		result := keepFirstNelements(test.inputMap, test.inputN)

		if !reflect.DeepEqual(result, test.expected) {
			t.Errorf("Test failed for inputMap: %v, expected: %v:, got: %v", test.inputMap, test.expected, result)
		}
	}
	tests1 := []struct {
		inputMap  map[string]int
		inputN    int
		expected  map[string]int
		expected1 map[string]int
	}{
		{
			inputMap:  map[string]int{"a": 10, "b": 20, "c": 30, "d": 40, "e": 50, "f": 10},
			inputN:    5,
			expected:  map[string]int{"e": 50, "d": 40, "c": 30, "b": 20, "a": 10},
			expected1: map[string]int{"e": 50, "d": 40, "c": 30, "b": 20, "f": 10},
		},
	}
	for _, test := range tests1 {
		result := keepFirstNelements(test.inputMap, test.inputN)

		if !(reflect.DeepEqual(result, test.expected) || reflect.DeepEqual(result, test.expected1)) {
			t.Errorf("Test failed for inputMap: %v, expected: %v:, got: %v", test.inputMap, test.expected, result)
		}
	}

}

func TestFitTransform(t *testing.T) {

	// word vectorizer
	documents := []string{
		"this is the first document",
		"this document is the second document",
		"and this is the third one",
		"is this the first document"}

	cv := &CountVectorizer{
		maxFeatures: 150000,
		nGramRange:  Range{1, 1},
		maxDf:       4,
		minDf:       1,
		norm:        false,
		analyzer:    "word",
		vocabulary:  nil,
	}

	x, err := cv.FitTransform(documents)
	if err != nil {
		t.Errorf("error: %f", err)
	}

	var desire = [][]float32{
		{0, 1, 1, 1, 0, 0, 1, 0, 1},
		{0, 2, 0, 1, 0, 1, 1, 0, 1},
		{1, 0, 0, 1, 1, 0, 1, 1, 1},
		{0, 1, 1, 1, 0, 0, 1, 0, 1},
	}

	real := make([][]float32, len(documents))

	for i := 0; i < len(documents); i++ {
		start := i * cv.maxFeatures
		end := (i + 1) * cv.maxFeatures
		real[i] = x[start:end]
	}

	if !reflect.DeepEqual(real, desire) {
		t.Errorf("Matrices are not the same want \n%v\ngot\n%v", desire, real)
	}

	cv = &CountVectorizer{
		maxFeatures: 150000,
		nGramRange:  Range{2, 2},
		maxDf:       4,
		minDf:       1,
		norm:        false,
		analyzer:    "word",
		vocabulary:  nil,
	}

	x, err = cv.FitTransform(documents)
	if err != nil {
		t.Errorf("error: %f", err)
	}

	desire = [][]float32{
		{0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0},
		{0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0},
		{1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0},
		{0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1}}

	for i := 0; i < len(documents); i++ {
		start := i * cv.maxFeatures
		end := (i + 1) * cv.maxFeatures
		real[i] = x[start:end]
	}

	if !reflect.DeepEqual(real, desire) {
		t.Errorf("Matrices are not the same want \n%v\ngot\n%v", desire, real)
	}

	// for i, v := range x {
	// 	fmt.Print(v)
	// 	fmt.Print(",")
	// 	if (i+1)%cv.maxFeatures == 0 {
	// 		fmt.Print("\n")
	// 	}
	// }

	// char vectorizer
	cv3 := &CountVectorizer{
		maxFeatures: 10,
		nGramRange:  Range{1, 2},
		maxDf:       4,
		minDf:       1,
		norm:        false,
		analyzer:    "char",
		vocabulary:  nil,
	}

	x3, err := cv3.FitTransform(documents)
	if err != nil {
		t.Errorf("error: %f", err)
	}

	desire3 := [][]float32{
		{4, 1, 2, 2, 3, 2, 3, 2, 4, 2},
		{5, 1, 4, 2, 2, 2, 3, 2, 4, 2},
		{5, 3, 2, 3, 3, 2, 2, 2, 3, 3},
		{4, 2, 2, 2, 3, 2, 3, 2, 4, 2},
	}

	for i := 0; i < len(documents); i++ {
		start := i * cv3.maxFeatures
		end := (i + 1) * cv3.maxFeatures
		real[i] = x3[start:end]
	}

	fmt.Println(cv3.vocabulary)
	if !reflect.DeepEqual(real, desire3) {
		t.Errorf("Matrices are not the same want \n%v\ngot\n%v", desire3, real)
	}

}

func BenchmarkKeepFirstNElements(b *testing.B) {
	inputMap := map[string]int{"a": 10, "b": 20, "c": 30, "d": 40, "e": 50, "f": 10}
	n := 5
	expected := map[string]int{"e": 50, "d": 40, "c": 30, "b": 20, "f": 10}

	for i := 0; i < b.N; i++ {
		result := keepFirstNelements(inputMap, n)
		if len(result) != len(expected) {
			b.Errorf("Result length is not equal to expected length")
		}
	}
}

func BenchmarkKeepFirstNElementsV0(b *testing.B) {
	inputMap := map[string]int{"a": 10, "b": 20, "c": 30, "d": 40, "e": 50, "f": 10}
	n := 5
	expected := map[string]int{"e": 50, "d": 40, "c": 30, "b": 20, "f": 10}

	for i := 0; i < b.N; i++ {
		result := keepFirstNelementsV0(inputMap, n)
		if len(result) != len(expected) {
			b.Errorf("Result length is not equal to expected length")
		}
	}
}

func BenchmarkFitTransform(b *testing.B) {
	documents := []string{
		"this is the first document",
		"this document is the second document",
		"and this is the third one",
		"is this the first document",
	}

	cv := &CountVectorizer{
		maxFeatures: 200,
		nGramRange:  Range{1, 1},
		maxDf:       4,
		minDf:       1,
		norm:        true,
		analyzer:    "char",
		vocabulary:  nil,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := cv.FitTransform(documents)
		if err != nil {
			b.Errorf("error: %f", err)
		}
	}
}

func TestCountVectorizer_CountVocab(t *testing.T) {
	cv := &CountVectorizer{
		maxFeatures: 10,
		nGramRange: Range{
			MinN: 1,
			MaxN: 1,
		},
		maxDf:      0,
		minDf:      0,
		norm:       false,
		analyzer:   "word",
		vocabulary: nil,
	}

	rawDocs := []string{
		"Hello world",
		"Hello Go",
		"Go programming",
	}

	expectedVocab := map[string]int{
		"Hello":       2,
		"world":       1,
		"Go":          2,
		"programming": 1,
	}

	actualVocab, _, err := cv.CountVocab(rawDocs)
	if err != nil {
		t.Errorf("error: %f", err)
	}

	if !reflect.DeepEqual(actualVocab, expectedVocab) {
		t.Errorf("CountVocab result does not match expected result")
	}
}
