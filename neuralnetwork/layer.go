package neuralnetwork

import "github.com/Jonny-Burkholder/neural-network/pkg/matrix"

//layer holds an abstract layer of neurons represented
//by a slice of inputs, and a layer activation function
type layer struct {
	Inputs     *matrix.Matrix
	Weights    *matrix.Matrix
	Biases     *matrix.Matrix
	Outputs    *matrix.Matrix
	Activation activation
}

//newLayer returns a new layer with a given number of inputs
//and outputs, and an activation function, and pseudo-random
//values for the weights and biases
func newLayer(inputs, outputs, activation int) *layer {
	return &layer{
		Inputs:     matrix.NewMatrix(inputs, 1, nil),
		Weights:    matrix.NewMatrix(inputs, outputs, nil),
		Biases:     matrix.NewMatrix(inputs, 1, nil),
		Outputs:    matrix.NewMatrix(outputs, 1, nil),
		Activation: getActivation(activation),
	}
}

//fire takes inputs and does the thing
func (l *layer) fire() { //return the outputs here?
	res := matrix.Dot(l.Inputs, l.Weights)
	l.Outputs = matrix.Add(res, l.Biases)
}

//stepBack takes in a slice representing the cost of the
//neural network. The function then finds the derivative of
//the cost with respect to the layer's activations and biases,
//respectively, and takes a small step towards the gradient's
//local minimum
func (l *layer) stepBack(cost, costPrime *matrix.Matrix) {
	//I'm not super worried about getting the calculus right here
	//just yet, just trying to get the structure back online
}
