package neuralnetwork

//layer holds an abstract layer of neurons represented
//by a slice of inputs, and a layer activation function
type layer struct {
	Inputs     []float64
	Weights    *Matrix
	Biases     *Vector
	Outputs    []float64
	Activation func(float64) float64
}

//newLayer returns a new layer with a given number of inputs
//and outputs, and an activation function, and pseudo-random
//values for the weights and biases
func newLayer(inputs, outputs, activation int) *layer {
	return &layer{
		Inputs:     make([]float64, inputs),
		Weights:    newMatrix(inputs, outputs),
		Biases:     newVector(inputs),
		Outputs:    make([]float64, outputs),
		Activation: getActivation(activation),
	}
}

//fire takes inputs and does the thing
func (l *layer) fire() { //return the outputs here?

}
