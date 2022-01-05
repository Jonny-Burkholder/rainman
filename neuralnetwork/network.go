package neuralnetwork

import "errors"

var nilNetwork = &Network{}
var nilHiddenLayers = []*layer{}

var errNoLayout = errors.New("Error: please input layout information for network layers")

//Network has an input and output layer, and
//a variadic number of hidden layers. The network
//forward-feeds inputs, and also regressively
//feeds backwards to train the weights and biases
type Network struct {
	Config       *Config
	InputLayer   *layer
	HiddenLayers []*layer
	OutputLayer  *layer
}

//NewNetwork takes a config struct and a variadic number of inputs
//representing layers. The minimum number of layers is two. Eventually
//this will have more nuanced options for things like activation type
func NewNetwork(config *Config, layout ...int) (*Network, error) {

	res := Network{
		Config: config,
	}

	if len(layout) < 1 {
		return nilNetwork, errNoLayout
	} else if len(layout) < 2 {
		res.InputLayer = newLayer(layout[0], layout[0], config.DefaultActivationType)
		res.OutputLayer = newLayer(layout[0], layout[0], config.OutputActivationType)
	} else {
		res.InputLayer = newLayer(layout[0], layout[1], config.DefaultActivationType)
		for i := 0; i < len(layout)-2; i++ { //for all the layers minus the first and last
			res.HiddenLayers[i] = newLayer(layout[i+1], layout[i+2], config.DefaultActivationType) //if this isn't clear, we can change the way it's indexed
		}
		//Outputlayer in this case is mostly just a transformation layer, so it will always have
		//the same number of inputs and outputs
		res.OutputLayer = newLayer(layout[len(layout)-1], layout[len(layout)-1], config.OutputActivationType)
	}
	return &res, nil
}
