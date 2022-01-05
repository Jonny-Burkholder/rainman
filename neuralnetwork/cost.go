package neuralnetwork

//meanSquared calculates a network cost by taking in a slice of predicted
//values and measuring them against a slice of expected values, squaring and
//summing each difference. It returns the cost and costPrime of the network
//for a given training input
//Even this I may change to take just one value, rather than a slice
func meanSquared(actual, expected []float64) ([]float64, []float64) {
	//crash if len(actual) != len(expected)? Who knows

	cost := make([]float64, len(actual))
	costPrime := make([]float64, len(actual))

	for i := 0; i < len(actual); i++ {
		a := actual[i] - expected[i]
		//I don't know if this is done all at once, or one example at a time
		cost[i] += a * a
		costPrime[i] += a * 2
	}
	return cost, costPrime
}
