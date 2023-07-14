package feature_extraction

import (
	"fmt"
	"testing"
)

func TestFitTransform(t *testing.T) {

	documents := []string{
		"This is the first document",
		"This document is the second document",
		"And this is the third one",
		"Is this the first document"}

	cv := &CountVectorizer{
		maxFeatures: 200,
		nGramRange:  Range{1, 1},
	}

	x := cv.FitTransform(documents)

	for i, v := range x {
		fmt.Print(v)
		fmt.Print(",")
		if i != 0 && i % cv.maxFeatures == 0{
			fmt.Print("\n")
		}
	}
}
