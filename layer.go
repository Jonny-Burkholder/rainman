package neuralnetwork

//Layer is a two-dimensional array of neurons. I guess. How does this work?
type Layer struct {
	Neurons [][]*Neuron //Should I abstract the 2-dimensionality? Like, just decide that a row is so many neurons, instead of using a real matrix?
}

//Activate does what it says on the tin
func (l *Layer) Activate()
