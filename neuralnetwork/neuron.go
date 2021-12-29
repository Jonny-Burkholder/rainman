package neuralnetwork

import (
	"errors"
	"math/rand"
	"time"
)

const (
	MinBias float64 = -10000
	MaxBias float64 = 10000
)

//This is only for human reference purposes. The Network will not use these
//Also I'll probably never get to implementing even half of these
var ActivationTypes []string = []string{"Sigmoid", "Relu", "Linear", "Binary-Step", "Tahn", "Leaky-Relu", "Swish", "Elu", "Gelu", "Selu"}

var errinvalidBias = errors.New("Bias is invalid")

//So we're going to use Neuron and Synapse interchangeably for now
type Neuron struct {
	Bias           float64
	ActivationType int //This should probably be done on a layer level, but oh well
}

//NewNeuron returns a new neuron with random values
func NewNeuron(activationType ...int) *Neuron {
	rand.Seed(time.Now().Unix())

	var a int

	if len(activationType) > 1 {
		if activationType[0] >= len(ActivationTypes) || activationType[0] < 0 {
			a = 0
		} else {
			a = activationType[0]
		}
	} else {
		//default activation type will be 0
		a = 0
	}

	return &Neuron{
		Bias:           rand.Float64(),
		ActivationType: a,
	}
}

//reWeight changes the weight for an individual neuron
func (n *Neuron) reBias(w float64) error {
	if w < MinBias || w > MaxBias {
		return errinvalidBias
	}
	n.Bias = w
	return nil
}

//Fire takes an input, runs it through the neuron, and spits out an output
//Note - I'm pretty sure on the layer level I can store the activation function
//as a parameter, instead of going through this switch statment every time. That
//should save us a lot of time
func (n *Neuron) Fire(a float64) float64 {
	var res float64
	//activate based on function
	switch n.ActivationType {
	case 1:
		res = Relu(a)
	case 2:
		res = Linear(a)
	case 3:
		res = BinaryStep(a)
	case 4:
		res = Tanh(a)
	case 5:
		res = LeakyRelu(a)
	case 6:
		res = Swish(a)
	case 7:
		res = Elu(a)
	case 8:
		res = Gelu(a)
	case 9:
		res = Selu(a)
	default:
		res = Sigmoid(a)
	}
	return res + n.Bias
}

//
