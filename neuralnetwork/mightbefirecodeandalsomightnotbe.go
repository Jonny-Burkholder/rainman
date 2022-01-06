package neuralnetwork

type nnn struct {
	o []float64
	x []float64
	w []float64
	h []float64
}

// nn is the network, in is the input data, tg is the target data, rate is the rate
func _backProp(nn *nnn, in, tg []float64, rate float64) {

	nhid := 3 //getNumberOfHiddenNeurons(nn)    // number of hidden neurons
	nops := 3 //nn.OutputLayer.Outputs.Length() // number of outputs
	nips := 4 //nn.InputLayer.Inputs.Length()   // number of inputs

	for i := 0; i < nhid; i++ {
		var sum float64
		// Calculate total error change with respect to output.
		for j := 0; j < nops; j++ {
			a := getPartialDerivativeOfErrorFunction(nn.o[j], tg[j])
			b := getPartialDerivativeOfActivationFunction(nn.o[j])
			sum += a * b * nn.x[j*nhid+i]
			// Correct weights in hidden to output layer.
			nn.x[j*nhid+i] -= rate * a * b * nn.h[i]
		}
		// Correct weights in input to hidden layer.
		for j := 0; j < nips; j++ {
			nn.w[i*nips+j] -= rate * sum * getPartialDerivativeOfActivationFunction(nn.h[i]) * in[j]
		}
	}
}

// getNumberOfHiddenNeurons is a helper function the return the number of hidden neurons
func getNumberOfHiddenNeurons(nn *Network) int {
	var size int
	for i := range nn.HiddenLayers {
		size += nn.HiddenLayers[i].Inputs.Length()
	}
	return size
}

// getPartialDerivativeOfErrorFunction returns partial derivative of error function--and can also be renamed
func getPartialDerivativeOfErrorFunction(a, b float64) float64 {
	return a - b
}

// getPartialDerivativeOfActivationFunction partial derivative of activation function--and can also be renamed
func getPartialDerivativeOfActivationFunction(a float64) float64 {
	return a * (1 - a)
}
