package feature_extraction

import (
	"math"
	"sort"
)

func normalize(s []float32) []float32 {
	var sum float32
	for _, v := range s {
		sum += v * v
	}
	sum = float32(math.Sqrt(float64(sum)))
	if sum != 0 {
		for i := range s {
			s[i] /= sum
		}
	}
	return s
}

func keepFirstNelementsV0(m map[string]int, n int) map[string]int {
	values := make([]int, len(m))
	for _, v := range m {
		values = append(values, v)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(values)))
	minValue := values[n-1]
	newVocab := make(map[string]int, n)

	for k, v := range m {
		if v >= minValue {
			newVocab[k] = v
		}
		if len(newVocab) == n {
			break
		}
	}
	return newVocab
}

type pair struct {
	key   string
	value int
}

func keepFirstNelements(inputMap map[string]int, n int) map[string]int {
	pairs := make([]pair, 0, len(inputMap))

	for k, v := range inputMap {
		pairs = append(pairs, pair{k, v})
	}

	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})

	expectedMap := make(map[string]int)

	for i := 0; i < n && i < len(pairs); i++ {
		expectedMap[pairs[i].key] = pairs[i].value
	}

	return expectedMap
}
