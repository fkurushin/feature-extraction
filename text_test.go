package feature_extraction

import (
	"fmt"
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
		maxDf:       1.0,
		minDf:       1,
	}

	x := cv.FitTransform(documents)

	for i, v := range x {
		fmt.Print(v)
		fmt.Print(",")
		if i != 0 && i%cv.maxFeatures == 0 {
			fmt.Print("\n")
		}
	}
}
