package ann

import (
	"log"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// aNN represents an artificial neural network
type aNN struct {
	config  *Config
	weights [][][]float64 // weights is the weights matrix of the network
	biases  [][]float64   // biases holds the biases of the network
	deltas  [][]float64   // deltas is the computed delta value for each layer
	zvalues [][]float64   // zvalues is the computed zvalues for each layer
	layers  int           // layers holds the number of layers in the network (excluding output layer)
	neurons []int         // neurons is the neurons per layer
	ActivationFn
}

func NewANN(conf *Config) *aNN {
	// check configuration
	checkConfig(conf)
	// initialize seed
	rand.Seed(conf.Seed)
	// setup new artificial neural network
	nn := aNN{
		config:       conf,
		layers:       len(conf.Layers) - 1,
		neurons:      conf.Layers[1:],
		ActivationFn: conf.Activation,
	}
	// if this works, leave it and if not, get rid of it
	if conf.IsRegression && nn.ActivationFn != Sigmoid {
		nn.ActivationFn = Sigmoid
	}
	// set up the weight matrices
	nn.weights = make([][][]float64, nn.layers)
	nn.weights[0] = make([][]float64, conf.Layers[0])
	for i := 0; i < conf.Layers[0]; i++ {
		nn.weights[0][i] = make([]float64, nn.neurons[0])
		for j := 0; j < nn.neurons[0]; j++ {
			nn.weights[0][i][j] = rand.Float64()
		}
	}
	for l := 1; l < nn.layers; l++ {
		nn.weights[l] = make([][]float64, nn.neurons[l-1])
		for i := 0; i < nn.neurons[l-1]; i++ {
			nn.weights[l][i] = make([]float64, nn.neurons[l])
			for j := 0; j < nn.neurons[l]; j++ {
				nn.weights[l][i][j] = rand.Float64()
			}
		}
	}

	// set up biases, deltas and z-value matrices
	nn.biases = make([][]float64, nn.layers)
	nn.deltas = make([][]float64, nn.layers)
	nn.zvalues = make([][]float64, nn.layers)
	for lyr := 0; lyr < nn.layers; lyr++ {
		nn.biases[lyr] = make([]float64, nn.neurons[lyr])
		for i := 0; i < nn.neurons[lyr]; i++ {
			nn.biases[lyr][i] = rand.Float64()
		}
		nn.deltas[lyr] = make([]float64, nn.neurons[lyr])
		nn.zvalues[lyr] = make([]float64, nn.neurons[lyr])
	}

	return &nn
}

func (nn *aNN) adjust(lyr, i, j int, rate float64) float64 {
	return -rate * nn.deltas[lyr][j] * nn.activate(lyr-1, i)
}

func (nn *aNN) activate(l, j int) float64 {
	return nn.ActFn(nn.zvalues[l][j])
}

func (nn *aNN) Train(data [][][]float64, iteration int, rate float64, debug bool) {
	for i := 0; i < iteration; i++ {
		for n := range data {
			nn.backProp(data[n][0], data[n][1], rate)
		}
		if debug {
			log.Printf("iteration %d\n", i)
		}
	}
}

func (nn *aNN) backProp(in, out []float64, rate float64) {

	// define zvalues
	_ = nn.feedForward(in)

	// define the output deltas
	last := len(nn.deltas) - 1
	for j := 0; j < len(nn.deltas[last]); j++ {
		nn.deltas[last][j] = (nn.activate(last, j) - out[j]) * nn.DerFn(nn.zvalues[last][j])
	}

	// define the inner deltas
	for lyr := len(nn.deltas) - 2; lyr >= 0; lyr-- {
		for j := 0; j < len(nn.deltas[lyr]); j++ {
			nn.deltas[lyr][j] = nn.delta(lyr, j)
		}
	}

	// update input weights
	for i := 0; i < len(nn.weights[0]); i++ {
		for j := 0; j < len(nn.weights[0][i]); j++ {
			nn.weights[0][i][j] += -rate * nn.deltas[0][j] * in[i]
		}
	}

	// update hidden weights
	for lyr := 1; lyr < len(nn.weights); lyr++ {
		for i := 0; i < len(nn.weights[lyr]); i++ {
			for j := 0; j < len(nn.weights[lyr][i]); j++ {
				nn.weights[lyr][i][j] += nn.adjust(lyr, i, j, rate)
			}
		}
	}

	// update biases
	for lyr := 0; lyr < len(nn.biases); lyr++ {
		for j := 0; j < len(nn.biases[lyr]); j++ {
			nn.biases[lyr][j] += -rate * nn.deltas[lyr][j]
		}
	}
}

// delta should only use in the back-propagation, otherwise it can return wrong value.
func (nn *aNN) delta(lyr, j int) float64 {
	var d float64
	for k := 0; k < nn.neurons[lyr+1]; k++ {
		d += nn.deltas[lyr+1][k] * nn.weights[lyr+1][j][k] * nn.DerFn(nn.zvalues[lyr][j])
	}
	return d
}

func (nn *aNN) feedForward(in []float64) []float64 {
	for lyr := 0; lyr < nn.layers; lyr++ {
		tmp := make([]float64, nn.neurons[lyr])
		for j := 0; j < nn.neurons[lyr]; j++ {
			nn.zvalues[lyr][j] = 0
			for i := 0; i < len(in); i++ {
				nn.zvalues[lyr][j] += nn.weights[lyr][i][j] * in[i]
			}
			nn.zvalues[lyr][j] += nn.biases[lyr][j]
			tmp[j] = nn.ActFn(nn.zvalues[lyr][j])
		}
		in = tmp
	}
	return in
}

func (nn *aNN) Predict(in []float64) []float64 {
	return nn.feedForward(in)
}

func (nn *aNN) String() string {
	return sprint3DMatrix(nn.weights)
}
