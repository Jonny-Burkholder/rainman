package neuralnetwork

import (
	"math/rand"
	"time"
)

//layer holds an abstract layer of neurons represented
//by a slice of inputs, and a layer activation function
type layer struct {
	Inputs     []float64
	Weights    [][]float64
	Biases     []float64
	Outputs    []float64
	Activation activation
}

//newLayer returns a new layer with a given number of inputs
//and outputs, and an activation function, and pseudo-random
//values for the weights and biases
func (n *Network) newLayer(inputs, outputs, activationType int) *layer {

	rand.Seed(time.Now().UnixNano())

	w := make([][]float64, inputs)

	for i := 0; i < inputs; i++ {
		w[i] = make([]float64, outputs)
		for j := 0; j < outputs; j++ {
			w[i][j] = rand.Float64()
		}
	}

	b := make([]float64, outputs)

	for i := 0; i < outputs; i++ {
		b[i] = rand.Float64()
	}

	var a activation

	if activationType != 1 {
		a = getActivation(activationType)
	} else {
		a = getActivation(activationType, n.Config.ReluLeak)
	}

	return &layer{
		Inputs:     make([]float64, inputs),
		Weights:    w,
		Biases:     b,
		Outputs:    make([]float64, outputs),
		Activation: a,
	}
}

//fire takes inputs and does the thing
func (l *layer) fire(input []float64) []float64 {
	//Yep, this is redundant, I know
	l.Inputs = input
	for i := 0; i < len(l.Inputs); i++ {
		for j := 0; j < len(l.Outputs); j++ {
			l.Outputs[j] += l.Inputs[i] * l.Weights[i][j]
		}
	}

	//I forget the formula, but I think this is right. Send the sum of inputs*weights
	//through the activation function, *then* add the bias
	for i := range l.Outputs {
		l.Outputs[i] = l.Activation.fire(l.Outputs[i]) + l.Biases[i]
	}

	return l.Outputs
}

//stepBack takes in a slice representing the cost of the
//neural network. The function then finds the derivative of
//the cost with respect to the layer's activations and biases,
//respectively, and takes a small step towards the gradient's
//local minimum
func (l *layer) stepBack(costPrime, rate float64) {
	l.updateWeights(costPrime, rate)
	l.updateBias(costPrime, rate)
}

//updateWeights - yep
func (l *layer) updateWeights(costPrime, rate float64) {
	//for each weight
	for i := 0; i < len(l.Weights); i++ {
		for j := 0; j < len(l.Weights[i]); j++ {
			nudge := (rate * costPrime * l.Inputs[i] * l.Activation.derivative(l.Outputs[j]))
			l.Weights[i][j] -= nudge
		}
	}
}

//updateBias
func (l *layer) updateBias(costPrime, rate float64) {
	for i, bias := range l.Biases {
		bias -= (rate * costPrime * l.Activation.derivative(l.Outputs[i]))
	}
}
