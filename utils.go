package feature_extraction

import "math"

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
