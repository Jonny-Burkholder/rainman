package neuralnetwork

import (
	"fmt"
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
	ErrorPrime []float64
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
			w[i][j] = rand.Float64()*2 - 1
		}
	}

	b := make([]float64, outputs)

	for i := 0; i < outputs; i++ {
		b[i] = rand.Float64()*2 - 1
	}

	var a activation

	if activationType != 1 {
		a = getActivation(activationType, n.Config.ReluLeak)
	} else {
		a = getActivation(activationType, n.Config.ReluLeak, n.Config.ReluCap)
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
	//zero your output, apparently this is important
	for k := 0; k < len(l.Outputs); k++ {
		l.Outputs[k] = 0
	}
	for i := 0; i < len(l.Inputs); i++ {
		for j := 0; j < len(l.Outputs); j++ {
			l.Outputs[j] += l.Inputs[i] * l.Weights[i][j]
		}
	}

	for i := range l.Outputs {
		l.Outputs[i] = l.Activation.fire(l.Outputs[i] + l.Biases[i])
	}

	return l.Outputs
}

//stepBack takes in a slice representing the cost of the
//neural network. The function then finds the derivative of
//the cost with respect to the layer's activations and biases,
//respectively, and takes a small step towards the gradient's
//local minimum. It returns a slice of errorPrime for the
//previous layer
func (l *layer) stepBack(rate float64, prime []float64) []float64 {
	l.ErrorPrime = prime
	new := l.newPrime()
	l.updateWeights(rate, prime)
	l.updateBias(rate, prime)
	return new
}

//updateWeights - yep
//this is confusing as crap
func (l *layer) updateWeights(rate float64, prime []float64) {
	for i := 0; i < len(prime); i++ {
		for j := 0; j < len(l.Inputs); j++ {
			nudge := (rate * prime[i] * l.Inputs[j] * l.Activation.derivative(l.Outputs[i]))
			l.Weights[j][i] -= nudge
		}
	}
}

//updateBias
func (l *layer) updateBias(rate float64, prime []float64) {
	for k := range l.Biases {
		l.Biases[k] -= rate * prime[k] * l.Activation.derivative(l.Outputs[k]) //eventually this will have to be z value
	}
}

//newPrime gives the error values for the layer one step back
//Ok this is obviously broken, since making the value smaller
//makes the network function better
func (l *layer) newPrime() []float64 {
	newPrime := make([]float64, len(l.Inputs))
	for i := 0; i < len(l.Outputs); i++ {
		for j := 0; j < len(l.Inputs); j++ {
			newPrime[j] += l.ErrorPrime[i] * l.Weights[j][i] * l.Inputs[j]
		}
	}
	//fmt.Println(newPrime)
	return newPrime
}

func (l *layer) String(w ...bool) string {
	s := ""

	if len(w) > 0 && w[0] == false {

	} else {

		s += "\nLayer Weights:\n"
		for i := 0; i < len(l.Weights); i++ {
			for j := 0; j < len(l.Weights[i]); j++ {
				s += fmt.Sprintf("%1.4f ", l.Weights[i][j])
			}
		}
	}

	s += "\nLayer Biases:\n"
	for i := 0; i < len(l.Biases); i++ {
		s += fmt.Sprintf("%1.4f ", l.Biases[i])
	}

	s += "\nLayer inputs:\n"
	for i := 0; i < len(l.Inputs); i++ {
		s += fmt.Sprintf("%1.4f ", l.Inputs[i])
	}

	s += "\nLayer outputs:\n"
	for i := 0; i < len(l.Outputs); i++ {
		s += fmt.Sprintf("%1.4f ", l.Outputs[i])
	}

	s += "\nLayer error:\n"
	for i := 0; i < len(l.ErrorPrime); i++ {
		s += fmt.Sprintf("%1.4f ", l.ErrorPrime[i])
	}

	s += "\n"

	return s
}
