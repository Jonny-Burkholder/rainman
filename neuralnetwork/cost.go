package neuralnetwork

type costfunction interface {
	Cost([]float64, []float64) (float64, float64)
}

func getCost(i int) costfunction {
	switch i {
	case 1:
		return &meanSquared{}
	case 2:
		return &crossEntropy{}
	default:
		return &meanSquared{}
	}
}

//meanSquared calculates a network cost by taking in a slice of predicted
//values and measuring them against a slice of expected values, squaring and
//summing each difference. It returns the partial derivative of the cost function
type meanSquared struct{}

//cost is the cost function of meanSquared
func (m meanSquared) Cost(actual, expected []float64) (float64, float64) {
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
		//I feel pretty confident this is the wrong algorithm for costPrime...
		//isn't is supposed to always be positive?
		cost += a * a
		costPrime += a * 2
	}
	return cost, costPrime
}

//crossEntropy. What does it do, you might ask? No clue
type crossEntropy struct{}

//cost is the cost function for crossEntropy
func (c *crossEntropy) Cost(actual, expected []float64) (float64, float64) {
	return 0, 0
}
