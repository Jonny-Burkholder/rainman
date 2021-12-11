package neuralnetwork

import "errors"

const (
	Minlength = 1
	Maxlength = 10 ^ 20 //I don't have any idea what this number is, but I know it's big
)

var errAlreadySized = errors.New("Input size already matches neural network")
var errInvalidSize = errors.New("Invalid size for neural network")

//Network will be a series of layers of neurons. Yeah
type Network struct {
	Layers []*Layer
	Size   int     //how many neurons are in the network. Int may be too small for this
	Bias   float32 //do I need this? No idea
}

//Compress takes input data that's too large for the network and compresses it into something
//more manageable. I'm assuming this will only be necessary for truly massive pieces of data,
//as the networks should, for the most part, be able to scale to their input
func (n *Network) Compress(inputs []float32) ([]float32, error) {
	if len(inputs) > n.Size {
		res := make([]float32, n.Size)
		//do stuff
		return res, nil
	}
	return []float32{}, errAlreadySized
}

//Upscale takes an input that's too small and interpolates intermediate values
func (n *Network) Upscale(inputs []float32) ([]float32, error) {
	if len(inputs) < n.Size {
		res := make([]float32, n.Size)
		//do magic
		return res, nil
	}
	return []float32{}, errAlreadySized
}

//Resize changes the size of the neural network, probably to match input size
func (n *Network) Resize(i int) error {
	if i < Minlength || i > Maxlength {
		return errInvalidSize
	} else if i == n.Size {
		return nil
	}
	n.Size = i
	//Do something to resize actual network, though that feels bad. What happens to all our weights?
	return nil
}
