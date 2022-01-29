package neuralnetwork

type costfunction interface {
	//predicted, expected
	Cost([]float64, []float64) ([]float64, []float64)
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
//I mean if I'm doing errors by neuron, not by layer, do I even need to
//square anything? I mean I guess so, since otherwise error prime is a
//constant, which is useless
func (m meanSquared) Cost(actual, expected []float64) ([]float64, []float64) {
	//crash if len(actual) != len(expected)? Who knows

	var length int

	if len(actual) > len(expected) {
		length = len(expected)
	} else {
		length = len(actual)
	}

	cost := make([]float64, length)
	prime := make([]float64, length)

	for i := 0; i < length; i++ {
		a := expected[i] - actual[i]
		//I feel pretty confident this is the wrong algorithm for costPrime...
		//isn't is supposed to always be positive?
		cost[i] = a * a
		prime[i] = -a
	}

	return cost, prime
}

//crossEntropy. What does it do, you might ask? No clue
type crossEntropy struct{}

//cost is the cost function for crossEntropy
func (c *crossEntropy) Cost(actual, expected []float64) ([]float64, []float64) {
	return nil, nil //huzzah for implicit pointer operations!
}
