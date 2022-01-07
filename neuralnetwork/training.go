package neuralnetwork

type TrainingData struct {
	inputs   []float64
	expected []float64
}

func (n *Network) Train(data *TrainingData) {
	n.ForwardFeed(data.inputs)
	n.Backpropagate(data.expected)
}
