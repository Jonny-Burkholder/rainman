package neuralnetwork

func dotProduct(input float64, weights []float64) float64 {
	var res float64

	for i := 0; i < len(weights); i++ {
		res += input * weights[i]
	}

	return res
}

func updateWeights(weights, updates [][]float64) {
	for i := 0; i < len(weights); i++ {
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] -= updates[i][j]
		}
	}
}

func updateBias(bias, updates []float64) {
	for i, v := range bias {
		v -= updates[i]
	}
}
