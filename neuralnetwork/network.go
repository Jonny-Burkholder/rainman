package neuralnetwork

//Network has an input and output layer, and
//a variadic number of hidden layers. The network
//forward-feeds inputs, and also regressively
//feeds backwards to train the weights and biases
type Network struct {
	Name         string
	ID           uint32
	Config       *Config
	InputLayer   *layer
	HiddenLayers []*layer
	OutputLayer  *layer
}
