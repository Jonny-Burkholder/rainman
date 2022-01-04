package neuralnetwork

var NilTrainingSet = &TrainingSet{}

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
	Cost []float64
}

//TrainingSet is a collection of training data. This is the data type that will
//by in large be used for the training of the network, as it will hold all
//of the necessary data for forward and backward propogation. It's also possible
//this could be handled from file instead of in memory, since we'll more than
//likely be dealing with large datasets
type TrainingSet struct {
	Data []*TrainingData
}

//LoadTrainingSet loads training data from csv or json
//it's a method because the data must match the network's
//inputs and ouputs. Which, now that I think about it, is a
//pretty glaring weakness in the network
func (n *Network) LoadTrainingSet(path string) (*TrainingSet, error) {
	return NilTrainingSet, nil
}

//Train takes in a training set and forward feeds and
//backpropogates is stochastically through the network. At least,
//that's the default until I learn other gradient descent methods.
//Honestly I'm starting to feel like the distinction between forward
//feeding and backpropogation is largely academic
func (n *Network) Train(t *TrainingSet) {
	var avgErr float64
	var iteration int
	for avgErr > n.Config.TrainingCondition && iteration <= n.Config.MaxSteps { //whichever comes first

	}
}
