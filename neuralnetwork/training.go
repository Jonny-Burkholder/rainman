package neuralnetwork

//TrainingInstance is a single instance of training data. So this might
//be a single photo of a cat, or a face, that is "labeled" with the expected
//network output
type TrainingInstance struct {
	Inputs   []float64
	Expected []float64 //expected is what we expect the output of the neural network to be
	//we could probably just give a single int for extpected, for the index of the output
	//neuron that we expect to be fully activated
}

//TrainingData is a collection of instances of a single type of data. So
//This would be a slice of pictures only of cats, or only of faces
type TrainingData struct {
	Data []*TrainingInstance
	Cost float64
}

//TrainingSet is a collection of training data. This is the data type that will
//by in large be used for the training of the network, as it will hold all
//of the necessary data for forward and backward propogation
type TrainingSet struct {
	Data []*TrainingData
}
