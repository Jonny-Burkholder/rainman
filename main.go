package main

import (
	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func main() {
	n, err := neuralnetwork.NewNetwork(neuralnetwork.DefaultConfig, 28*28, 24, 10)

	if err != nil {
		panic(err)
	}

	n.TrainMnist()
	n.TestMnist()

}
