package feature_extraction

import (
	"reflect"
	"testing"
)

func TestFitTransform(t *testing.T) {

	documents := []string{
		"this is the first document",
		"this document is the second document",
		"and this is the third one",
		"is this the first document"}

	cv := &CountVectorizer{
		maxFeatures: 200,
		nGramRange:  Range{1, 1},
		maxDf:       4,
		minDf:       1,
		norm:        false,
	}

	x := cv.FitTransform(documents)

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

	// for i, v := range x {
	// 	fmt.Print(v)
	// 	fmt.Print(",")
	// 	if (i+1)%cv.maxFeatures == 0 {
	// 		fmt.Print("\n")
	// 	}
	// }
}


