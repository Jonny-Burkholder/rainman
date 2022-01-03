package neuralnetwork

//meanSquared does the whole, you know, mean
//squared cost thing. Not really sure this
//needs to be a method at all. Anyway, it takes
//in two slices, the actual and expected values for
//a layer of neurons, and it spits out not only the
//cost for each of those values, but the derivative as well
func (n *Network) meanSquared(actual, expected []float64) ([]float64, []float64) {
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
