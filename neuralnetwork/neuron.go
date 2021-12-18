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

//So we're going to use Neuron and Synapse interchangeably for now
type Neuron struct {
	Weight float32
	Next   []*Neuron //Neurons in the next layer that
}

//NewNeuron returns a new neuron with random values
func NewNeuron(length int) *Neuron {
	rand.Seed(time.Now().Unix())

	return &Neuron{
		Weight: rand.Float32(),
	}
}

//reWeight changes the weight for an individual neuron
func (n *Neuron) reWeight(w float32) error {
	if w < MinBias || w > MaxBias {
		return errinvalidBias
	}
	n.Weight = w
	return nil
}

//Fire takes an input, runs it through the neuron, and spits out an output
func (n *Neuron) Fire(a float32) float32 {
	return a * n.Weight
}
