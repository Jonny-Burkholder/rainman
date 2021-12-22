package neuralnetwork

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

const (
	MinSize   = 1
	MaxSize   = 10 ^ 12 //One Trillion should be enough neurons per layer, yeah?
	MinLayers = 2
	MaxLayers = 1000 //This is completely arbitrary and probably way too much
)

var errAlreadySized = errors.New("Input size already matches neural network")
var errInvalidSize = errors.New("Invalid size for neural network")
var errInvalidLayers = errors.New("Invalid number of layers")
var nilNetwork = &Network{}

//Network will be a series of layers of neurons. Yeah
type Network struct {
	Name        string
	ID          float64
	Config      *Config
	CurrentStep float64 //the current step size, after adjusting for learning rate, etc
	Layers      []*Layer
	Size        int //how many neurons are in the network. Int may be too small for this
	Bias        float64
	Clusters    []*Cluster          //Clusters represent information that is yet uncategorized, but is clustered together
	Children    []*Network          //To pass along for more specialized recognition
	OutPuts     map[int]interface{} //Terrible! Just terrible
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
		res[i] = NewLayer(neurons[i], neurons[i+1])
		size += neurons[i]
	}

	//I should probably have an "OutputLayer" type, instead of doing this
	res[len(res)-1] = NewLayer(len(neurons)-1, 0)

	size += neurons[len(neurons)-1]

	rand.Seed(time.Now().UnixNano())

	return &Network{
		Name:        name,
		ID:          rand.Float64(),
		Config:      config,
		CurrentStep: config.BaseStepSize,
		Layers:      res,
		Size:        size,
		Bias:        rand.Float64(),
	}, nil
}

//Compress takes input data that's too large for the network and compresses it into something
//more manageable. I'm assuming this will only be necessary for truly massive pieces of data,
//as the networks should, for the most part, be able to scale to their input
func (n *Network) Compress(inputs []float64) ([]float64, error) {
	if len(inputs) > n.Size {
		res := make([]float64, n.Size) //n.Size does *not* work here. We'll have to add another argument to the function
		//do stuff
		return res, nil
	}
	return []float64{}, errAlreadySized
}

//Upscale takes an input that's too small and interpolates intermediate values
func (n *Network) Upscale(inputs []float64) ([]float64, error) {
	if len(inputs) < n.Size {
		res := make([]float64, n.Size)
		//do magic
		return res, nil
	}
	return []float64{}, errAlreadySized
}

//Resize changes the size of the neural network, probably to match input size
func (n *Network) Resize(i int) error {
	//No idea how this will work, or if it's even a good idea
	return nil
}

//Activate takes a slice of input data and passes it through each layer of the network
func (n *Network) Activate(input []float64) float64 {
	//activate each layer
	return 0
}

//StepSize takes a slope as an input, and returns a step size
func (n *Network) StepSize(s float64) float64 {
	return s * n.Config.LearningRate
}

//Stochastic takes an integer l, and returns a slice of indeces bounded between zero and l
//The resultant slice is the fixed length of data points used for stochastic gradient descent,
//and is used to
func (n *Network) Stochastic(l int) []int {
	rand.Seed(time.Now().UnixNano())
	temp := make(map[int]bool)
	res := make([]int, n.Config.Stochastic)
	for i := 0; i < l; i++ {
		//if the number is already in use, loop until a unique number is reached
		for {
			num := rand.Intn(l)
			if _, ok := temp[num]; ok == !true {
				res[i] = num
				temp[num] = true
				break
			}
		}
	}
	return res
}

//Descend does the least squares gradient descent thing
//I don't actually know how to do this yet
func (n *Network) Descend(output, expected []float64) {}
