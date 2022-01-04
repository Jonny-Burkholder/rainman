package neuralnetwork

import (
	"math/rand"
	"time"
)

var LayerTypes []string = []string{"Synapse", "Hidden", "Output"}
var nilLayer = &Layer{}

//Layer is a two-dimensional array of neurons. I guess. How does this work?
type Layer struct {
	LayerType      int
	ActivationType int
	Neurons        []*Neuron
	Weights        [][]float64 //Matrix of connections to next layer
	Output         []float64   //need some persistence here to backpropogate
}

//NewLayer takes two inputs, nsize and wsize, where nsize is the number of neurons
//in the layer, and wsize is the number of weights for each neuron, aka the number
//of neurons in the next layer. The function returns a new layer
func NewLayer(nsize, wsize, ltype, atype int) *Layer {

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
		LayerType:      ltype,
		ActivationType: atype,
		Neurons:        n,
		Weights:        w,
		Output:         make([]float64, wsize),
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
			l.Output[i] = res[i]
		}
		return res
	}
}

//ActivatePrime does the derivative thing. Sorry, I'm tired and in a hurry
func (l *Layer) ActivatePrime(z float64) float64 {
	var res float64
	switch l.ActivationType {
	//not going to do the tedious work of inputing these all right now
	default:
		res = LeakyRelu6Prime(z, .001)
	}
	return res
}

//WeighPrimes takes a slice of float64 as its input. This slice represents
//the derivative of the cost of each neuron in this layer, which has
//already been calculated elsewhere. The function returns a slice of
//float64 representing a hint, if you will, of how much to nudge the
//weights
func (l *Layer) WeightPrime(costPrime []float64, previous *Layer) [][]float64 { //first return is weight, second return is bias
	//res represents... something. a slice of somethings
	res := make([][]float64, len(previous.Neurons))

	//for each weight in the previous layer, which we can break down as
	//for each neuron in the last layer
	for i := 0; i < len(previous.Neurons); i++ {
		//working just like we did in the NewLayer function
		res[i] = make([]float64, len(l.Neurons))
		//for each output associated with that neuron...
		for j := 0; j < len(l.Neurons); j++ {
			z := previous.Output[j]
			aPrime := l.ActivatePrime(z)
			//derivative of the cost of the current neuron output times
			//derivative of the activation of the current neuron times
			//the derivative of the input of the current neuron
			//j is the current neuron of the current layer
			res[i][j] += costPrime[j] * aPrime * z
		}
	}
	return res
}

//BiasPrime does the same thing as WeightPrime, but for Biases. And also on the current
//layer, not the previous one? I think
func (l *Layer) BiasPrime(costPrime []float64, previous *Layer) []float64 {
	res := make([]float64, len(l.Neurons))

	for i := 0; i < len(l.Neurons); i++ {
		res[i] = costPrime[i] * l.ActivatePrime(previous.Output[i])
	}
	return res
}

//Adjust takes in two slices, which are the derivative of the weights
//and biases of the layer, and a float, which is the learning rate. The
//weights and biases are then adjusted by subtracting their derivative
//multiplied by the learning rate
func (l *Layer) Adjust(weightPrime [][]float64, biasPrime []float64, rate float64) error { //gut check says there should be an error here

	for i := 0; i < len(l.Neurons); i++ {
		l.Neurons[i].reBias(biasPrime[i] * rate)
		for j := 0; j < len(l.Weights[i]); j++ {
			l.Weights[i][j] -= (weightPrime[i][j] * rate)
		}
	}

	return nil
}

//Descend puts all the pieces together. It also needs to make a return
//so that that can be input for the next layer, but I don't know how
//to do that
func (l *Layer) Descend(cost, costPrime []float64, previous *Layer, rate float64) {
	weightPrime := l.WeightPrime(cost, previous)
	biasPrime := l.BiasPrime(costPrime, previous)
	l.Adjust(weightPrime, biasPrime, rate)
	//so I don't think we return anything here, I think we just use the overall network cost every time
}
