package neuralnetwork

import (
	"math/rand"
	"time"
)

var LayerTypes []string = []string{"Synapse", "Hidden", "Output"}
var nilLayer = &Layer{}

//Layer is a two-dimensional array of neurons. I guess. How does this work?
type Layer struct {
	LayerType int
	Neurons   []*Neuron
	Weights   [][]float64 //Matrix of connectins to next layer
}

//NewLayer takes two inputs, nsize and wsize, where nsize is the number of neurons
//in the layer, and wsize is the number of weights for each neuron, aka the number
//of neurons in the next layer. The function returns a new layer
func NewLayer(nsize, wsize, ltype int) *Layer {

	if ltype > len(LayerTypes) {
		//return an error
	}

	//activation type should be handled here, not at the neuron level

	rand.Seed(time.Now().UnixNano())

	n := make([]*Neuron, nsize)
	for i := 0; i < nsize; i++ {
		n[i] = NewNeuron()
	}

	//each row of the weights matrix will represent a neuron in the current layer
	w := make([][]float64, nsize)
	for i := 0; i < nsize; i++ {
		//and each column in each row will represent a neuron in the current layer's
		//relationship to a neuron in the next layer
		w[i] = make([]float64, wsize)
		for j := 0; j < wsize; j++ {
			//rando-fill those weights
			w[i][j] = rand.Float64()
		}
	}

	return &Layer{
		LayerType: ltype,
		Neurons:   n,
		Weights:   w,
	}

}

//Activate does a nasty matrix multiplication thing
//Evemtially I'll bring in an optimized matrix library,
//but for now I'll hand-roll it
func (l *Layer) Activate(inputs []float64) []float64 {
	if l.LayerType != 2 {
		res := make([]float64, len(l.Weights[0])) //there's gotta be a more clear way to do this
		//Send each input through its respective neuron
		for i := 0; i < len(inputs); i++ {
			a := l.Neurons[i].Fire(inputs[i])
			//Feed the output from each neuron through the weights matrix and =+ the
			//dot product to the jth index of res
			for j := 0; j < len(res); j++ {
				res[j] += l.Weights[i][j] * a
			}
		}
		//return that vector
		return res
	} else {
		res := make([]float64, len(inputs))
		for i := 0; i < len(inputs); i++ {
			res[i] = l.Neurons[i].Fire(inputs[i])
		}
		return res
	}
}

//WeightPrime takes a slice of float64 as its input. This slice represents
//the derivative of the cost of each neuron in this layer, which has
//already been calculated elsewhere. The function returns a slice of
//float64 representing a hint, if you will, of how much to nudge the cost
//associated with this slice of relationships
func (l *Layer) WeightPrime(costPrime []float64) []float64 {
	//res represents... something. a slice of somethings
	res := make([]float64, len(l.Neurons))

	//for each neuron
	for i, n := range l.Neurons {
		//for each weight being input into the neuron
		//costprime[i] * n.activationPrime * z(l)prime
	}
	return res
}
