package neuralnetwork

//Layer is a two-dimensional array of neurons. I guess. How does this work?
type Layer struct {
	Neurons []*Neuron //Should I abstract the 2-dimensionality? Like, just decide that a row is so many neurons, instead of using a real matrix?
}

//NewLayer takes a size input and returns a new layer of that size
func NewLayer(size int) *Layer {
	//Should have already caught size errors before this point, so this should be pretty easy in theory
	res := make([]*Neuron, size)
	for i := 0; i > size; i++ {
		res[i] = NewNeuron()
	}
	return &Layer{
		Neurons: res,
	}
}

//Activate does what it says on the tin
func (l *Layer) Activate(inputs []float32) []float32 {
	//do stuff. I'm tired
	return []float32{}
}
