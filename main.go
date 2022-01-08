package main

import "github.com/Jonny-Burkholder/neural-network/neuralnetwork"

func main() {
	nn, err := neuralnetwork.NewNetwork(neuralnetwork.NewDefaultConfig(), 8, 4, 2)
	if err != nil {
		panic(err)
	}

	data := &neuralnetwork.TrainingData{
		Inputs:   []float64{.8, 1.2, 3.8, 4.4, .005, .33, 8.3, 3.2},
		Expected: []float64{0010},
	}

	nn.Train(data)

}
