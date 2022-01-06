package neuralnetwork

import "github.com/Jonny-Burkholder/neural-network/pkg/matrix"

//meanSquared calculates a network cost by taking in a slice of predicted
//values and measuring them against a slice of expected values, squaring and
//summing each difference. It returns the cost and costPrime of the network
//for a given training input
//Even this I may change to take just one value, rather than a slice
func meanSquared(actual, expected *matrix.Matrix) (*matrix.Matrix, *matrix.Matrix) {
	//crash if len(actual) != len(expected)? Who knows

	cost := matrix.NewMatrix(actual.Length(), 1, nil)
	costPrime := matrix.NewMatrix(actual.Length(), 1, nil)

	for i := 0; i < actual.Length() && i < expected.Length(); i++ {
		a := actual.GetAt(i, 1) - expected.GetAt(i, 1)
		//I don't know if this is done all at once, or one example at a time
		cost.SetAt(i, 1, (a * a))
		costPrime.SetAt(i, 1, (a * 2))
	}
	return cost, costPrime
}
