package neuralnetwork

import "fmt"

type TrainingData struct {
	Inputs   []float64
	Expected []float64
}

func (n *Network) Train(data *TrainingData) {
	n.ForwardFeed(data.Inputs)
	fmt.Printf("Network before training:\n\n%s", n.String())
	n.Backpropagate(data.Expected)
	fmt.Printf("Network after training:\n\n%s", n.String())
}
