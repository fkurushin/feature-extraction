package feature_extraction

import (
	"reflect"
	"testing"
)

func TestAnalyze(t *testing.T) {
	cv := &CountVectorizer{
		maxFeatures: 2,
		nGramRange:  Range{1, 2},
	}
	analyzed := cv.Analyze("платье женское")
	arr := []string{"п", "пл", "л", "ла", "а", "ат", "т", "ть", "ь", "ье", "е", "е ", " ", " ж", "ж", "же", "е", "ен", "н", "нс", "с", "ск", "к", "ко", "о", "ое", "е"}

	if !reflect.DeepEqual(arr, analyzed) {
		t.Errorf("\nexpected:\n%#v got:\n%#v", analyzed, arr)
	}

	cv = &CountVectorizer{
		maxFeatures: 2,
		nGramRange:  Range{1, 3},
	}
	analyzed = cv.Analyze("платье женское")
	arr = []string{"п", "пл", "пла", "л", "ла", "лат", "а", "ат", "ать", "т", "ть", "тье", "ь", "ье", "ье ", "е", "е ", "е ж", " ", " ж", " же", "ж", "же", "жен", "е", "ен", "енс", "н", "нс", "нск", "с", "ск", "ско", "к", "ко", "кое", "о", "ое", "е"}

	if !reflect.DeepEqual(arr, analyzed) {
		t.Errorf("\nexpected:\n%#v got:\n%#v", analyzed, arr)
	}

}

func TestLimitFeatures(t *testing.T) {
	cv := &CountVectorizer{
		maxFeatures: 2,
	}
	vocab := map[string]int{
		"п":  2,
		"пл": 3,
		"л":  1,
	}
	expected := map[string]int{
		"п":  2,
		"пл": 3,
	}
	result := cv.LimitFeatures(vocab)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Test case 1 failed: expected %v but got %v", expected, result)
	}

	cv = &CountVectorizer{
		maxFeatures: 5,
	}
	vocab = map[string]int{
		"п":  10,
		"пл": 5,
		"ла": 3,
		"ат": 2,
		"т":  1,
	}
	expected = map[string]int{
		"п":  10,
		"пл": 5,
		"ла": 3,
		"ат": 2,
		"т":  1,
	}
	result = cv.LimitFeatures(vocab)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Test case 2 failed: expected %v but got %v", expected, result)
	}

	cv = &CountVectorizer{
		maxFeatures: 0,
	}
	vocab = map[string]int{
		"п":  10,
		"пл": 5,
		"ла": 3,
		"ат": 2,
		"т":  1,
	}
	expected = map[string]int{}
	result = cv.LimitFeatures(vocab)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Test case 3 failed: expected %v but got %v", expected, result)
	}

	cv = &CountVectorizer{
		maxFeatures: 3,
	}
	vocab = map[string]int{
		"п":  10,
		"пл": 5,
		"ла": 3,
		"за": 3,
		"ра": 3,
		"ат": 2,
		"т":  1,
	}
	expected = map[string]int{
		"п":  10,
		"пл": 5,
		"ла": 3,
	}
	result = cv.LimitFeatures(vocab)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Test case 3 failed: expected %v but got %v", expected, result)
	}

}
