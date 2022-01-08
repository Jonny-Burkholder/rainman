package neuralnetwork

type TrainingData struct {
	Inputs   []float64
	Expected []float64
}

func (n *Network) Train(data *TrainingData) {
	n.ForwardFeed(data.Inputs)
	n.Backpropagate(data.Expected)
}
