package convolution

//Convolution is a convolutional neural network
//May even be able to just define this type as
//a slice of layers. Maybe not though
type Convolution struct {
	Layers []*layer `json:"layers"`
}

//NewConvolution returns a new convolutional neural
//network based on a config file. This network can
//either be fully instantiated with randomized values,
//or can be completely blank
func NewConvolution(config *Config) *Convolution{
	//TODO: actually make this
	return &Convolution{}
}


