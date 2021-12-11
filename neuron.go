package neuralnetwork

import (
	"errors"
	"math/rand"
	"time"
)

const (
	Minweight float32 = -10000
	Maxweight float32 = 10000
)

var errinvalidweight = errors.New("Weight is invalid")

type Neuron struct {
	Weight float32 //I have no idea what I'm doing
}

//NewNeuron returns a new neuron with a random value
func NewNeuron() *Neuron {
	rand.Seed(time.Now().Unix())
	return &Neuron{
		Weight: rand.Float32(), //This will need to be scaled to something between 1 and 0, obviously
	}
}

//reWeight changes the weight for an individual neuron
func (n *Neuron) reWeight(w float32) error {
	if w < Minweight || w > Maxweight {
		return errinvalidweight
	}
	n.Weight = w
	return nil
}

//Activate takes an input, runs it through the neuron, and spits out an output
//This probably shouldn't be done at the neuron level, but what the heck, I'll
//Make it better in a refactor down the road
func (n *Neuron) Activate(a float32) float32 {
	return a * n.Weight
}
