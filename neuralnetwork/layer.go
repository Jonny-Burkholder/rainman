package neuralnetwork

//layer holds an abstract layer of neurons represented
//by a slice of inputs, and a layer activation function
type layer struct {
	Inputs     *Vector
	Weights    *Matrix
	Biases     *Vector
	Outputs    *Vector
	Activation func(float64) float64
}

//newLayer returns a new layer with a given number of inputs
//and outputs, and an activation function, and pseudo-random
//values for the weights and biases
func newLayer(inputs, outputs, activation int) *layer {
	return &layer{
		Inputs:     NewVector(inputs),
		Weights:    NewMatrix(inputs, outputs),
		Biases:     NewVector(inputs),
		Outputs:    NewVector(outputs),
		Activation: getActivation(activation),
	}
}

//fire takes inputs and does the thing
func (l *layer) fire() { //return the outputs here?
	res := *DotProduct(l.Inputs, l.Weights)
	l.Outputs = AddMatrix(res, l.Biases)
}

//stepBack takes in a slice representing the cost of the
//neural network. The function then finds the derivative of
//the cost with respect to the layer's activations and biases,
//respectively, and takes a small step towards the gradient's
//local minimum
func (l *layer) stepBack(cost []float64) {
	//I'm not super worried about getting the calculus right here
	//just yet, just trying to get the structure back online
}
