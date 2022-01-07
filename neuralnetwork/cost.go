package neuralnetwork

//meanSquared calculates a network cost by taking in a slice of predicted
//values and measuring them against a slice of expected values, squaring and
//summing each difference. It returns the cost and costPrime of the network
//for a given training input
//Even this I may change to take just one value, rather than a slice
func meanSquared(actual, expected []float64) (float64, float64) {
	//crash if len(actual) != len(expected)? Who knows

	var length int

	if len(actual) > len(expected) {
		length = len(expected)
	} else {
		length = len(actual)
	}

	var cost, costPrime float64

	for i := 0; i < length; i++ {
		a := expected[i] - actual[i]
		//I don't know if this is done all at once, or one example at a time
		cost += a * a
		costPrime += a * 2
	}
	return cost, costPrime
}
