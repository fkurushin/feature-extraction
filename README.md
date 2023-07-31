# feature-extraction

go-sklearn, feature-extraction module
go package for sklearn feature extraction module

```
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
		panic(err)
	}

```
options: analyzer: "char" or "word"
