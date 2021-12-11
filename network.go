package neuralnetwork

//Network will be a series of layers of neurons. Yeah
type Network struct {
	Layers []*Layer
	Bias   float32 //do I need this? No idea
}
