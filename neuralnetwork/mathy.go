package neuralnetwork

func dotProduct(input float64, weights []float64) float64 {
	var res float64

	for i := 0; i < len(weights); i++ {
		res += input * weights[i]
	}

	return res
}
