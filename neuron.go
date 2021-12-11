package neuralnetwork

import "errors"

const (
	Minweight float32 = -10000
	Maxweight float32 = 10000
)

var errinvalidweight = errors.New("Weight is invalid")

type Neuron struct {
	Weight float32 //I have no idea what I'm doing
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
