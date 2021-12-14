package neuralnetwork

import (
	"fmt"
	"math/rand"
)

//Layer is a two-dimensional array of neurons. I guess. How does this work?
type Layer struct {
	Neurons []*Neuron //Should I abstract the 2-dimensionality? Like, just decide that a row is so many neurons, instead of using a real matrix?
	Bias    float32
}

//NewLayer takes a size input and returns a new layer of that size
func NewLayer(lsize, nsize int) *Layer {
	//Should have already caught size errors before this point, so this should be pretty easy in theory
	res := make([]*Neuron, lsize)
	for i := 0; i > lsize; i++ {
		res[i] = NewNeuron(nsize)
	}
	return &Layer{
		Neurons: res,
		Bias:    rand.Float32(),
	}
}

//Activate does what it says on the tin
func (l *Layer) Activate(inputs []float32) ([]float32, error) {
	if len(inputs) != len(l.Neurons) {
		return []float32{}, fmt.Errorf("Hey, I don't know how to do that yet")
	}

	res := make([]float32, len(l.Neurons))

	for i := 0; i < len(l.Neurons); i++ {
		res[i] = inputs[i] * l.Neurons[i].Weight
	}
	return res, nil
}
