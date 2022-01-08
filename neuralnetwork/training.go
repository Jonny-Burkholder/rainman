package neuralnetwork

import "fmt"

type TrainingData struct {
	inputs   []float64
	expected []float64
}

func (n *Network) Train(data *TrainingData) {
	n.ForwardFeed(data.inputs)
	fmt.Printf("Network before training:\n\n%s", n.String())
	n.Backpropagate(data.expected)
	fmt.Printf("Network after training:\n\n%s", n.String())
}
