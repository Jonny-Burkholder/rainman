package neuralnetwork

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

const (
	MinSize   = 1
	MaxSize   = 1000000000000 //One Trillion should be enough neurons per layer, yeah?
	MinLayers = 2
	MaxLayers = 1000 //This is completely arbitrary and probably way too much
)

var errAlreadySized = errors.New("Input size already matches neural network")
var errInvalidSize = errors.New("Invalid size for neural network")
var errInvalidLayers = errors.New("Invalid number of layers")
var nilNetwork = &Network{}

//Network will be a series of layers of neurons. Yeah
type Network struct {
	Name   string
	ID     float64
	Config *Config
	//CurrentStep float64 //the current step size, after adjusting for learning rate, etc
	InputLayer  *Layer
	Layers      []*Layer
	OutputLayer *Layer
	Size        int //how many neurons are in the network. Int may be too small for this
	//we're going to forget about this network feeding forward into child networks for now
	//and just focus on getting it working
}

//NewNetwork takes a config file and several integers as arguments, and produces a neural
//network with a random bias. The each index in the neurons argument will be a new layer
//for the network, with the integer at that index representing the number of neurons at that
//layer. So an argument of 16, 8, 4, 1 will create a new network with 16 input neurons, two
//hidden layers, and an output layer containing a single neuron
func NewNetwork(name string, config *Config, neurons ...int) (*Network, error) {
	if len(neurons) > MaxLayers || len(neurons) < MinLayers {
		return nilNetwork, errInvalidLayers
	}

	var size int

	res := make([]*Layer, len(neurons))
	for i := 0; i < len(res)-1; i++ {
		if neurons[i] > MaxSize || neurons[i] < MinSize {
			return nilNetwork, fmt.Errorf("Invalid size: Layer at index %v", i)
		}
		if i != 0 {
			//if it's not the first or last layer, layer type will be 1, or "hidden"
			res[i] = NewLayer(neurons[i], neurons[i+1], 1, config.ActivationType)
		} else {
			//if it *is* the first layer, then layer type will be 0, or "synapse"
			res[i] = NewLayer(neurons[i], neurons[i+1], 0, config.ActivationType)
		}
		size += neurons[i]
	}

	//I should probably have an "OutputLayer" type, instead of doing this
	res[len(res)-1] = NewLayer(neurons[len(neurons)-1], 0, 2, config.OutputActivationType)

	size += neurons[len(neurons)-1]

	rand.Seed(time.Now().UnixNano())

	return &Network{
		Name:   name,
		ID:     rand.Float64(),
		Config: config,
		Layers: res,
		Size:   size,
	}, nil
}

//Resize changes the size of the neural network, probably to match input size
//Even though, you know, that makes no freaking sense
func (n *Network) Resize(i int) error {
	//No idea how this will work, or if it's even a good idea
	return nil
}

//Activate takes a slice of inputs and sends the activations
//layer by layer through the network, until the output is reached
func (n *Network) Activate(a []float64) ([]float64, error) {
	if len(a) != len(n.Layers[0].Neurons) {
		return []float64{}, fmt.Errorf("Invalid number of inputs: want %v, got %v", len(n.Layers[0].Neurons), len(a))
	}

	var res []float64
	res = append(res, a...)

	//chain the activations down through the layers
	for i := 0; i < len(n.Layers); i++ {
		res = n.Layers[i].Activate(res)
	}
	return res, nil
}

//Descend does the least squares gradient descent thing
//I don't actually know how to do this yet
func (n *Network) BackPropogate(cost, costPrime []float64) {
	for i := len(n.Layers) - 1; i > 0; i-- {
		n.Layers[i].Descend(cost, costPrime, n.Layers[i-1], n.Config.LearningRate)
	}
}

//String is a stringer function for a network
func (n *Network) String() string {
	var s string
	s += fmt.Sprintf("Network Name: %s\n", n.Name)
	s += fmt.Sprintf("%v layers:\n", len(n.Layers))
	for i, n := range n.Layers {
		s += fmt.Sprintf("	Layer %v: [%v] neurons, [%v] connections\n", i, len(n.Neurons), len(n.Weights)*len(n.Weights[0]))
	}
	return s
}
