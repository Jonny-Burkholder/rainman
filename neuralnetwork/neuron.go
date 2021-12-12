package neuralnetwork

import (
	"errors"
	"math/rand"
	"time"
)

const (
	MinBias float32 = -10000
	MaxBias float32 = 10000
)

var errinvalidBias = errors.New("Bias is invalid")

type Neuron struct {
	Synapses   []float32 //we really only need the parameters, right?
	Parameters []float32
	Bias       float32
}

//NewNeuron returns a new neuron with random values
func NewNeuron(length int) *Neuron {
	rand.Seed(time.Now().Unix())

	s := make([]float32, length)
	p := make([]float32, length)

	//these numbers will need to be normalized to numbers between 0 and 1. I think?
	for i := 0; i < length; i++ {
		s[i] = rand.Float32()
		p[i] = rand.Float32()
	}
	return &Neuron{
		Synapses:   s,
		Parameters: p,
		Bias:       rand.Float32(),
	}
}

//reBias changes the Bias for an individual neuron
func (n *Neuron) reBias(w float32) error {
	if w < MinBias || w > MaxBias {
		return errinvalidBias
	}
	n.Bias = w
	return nil
}

//Activate takes an input, runs it through the neuron, and spits out an output
//This probably shouldn't be done at the neuron level, but what the heck, I'll
//Make it better in a refactor down the road
func (n *Neuron) Activate(a []float32) float32 {
	//first, we create a variable to hold the result
	var res float32
	//next, we multiply each input by its corresponding parameter, or weight, and
	//add that to the res
	l := len(n.Parameters)
	for i := 0; i < len(a); i++ {
		if i >= l { //this is not the right way to handle this. Data should be compressed to fit NN size
			break
		}
		res += a[i] * n.Parameters[i]
	}
	return res + n.Bias
}
