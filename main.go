package main

import (
	"fmt"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func main() {
	n, err := neuralnetwork.NewNetwork(neuralnetwork.DefaultConfig, 28*28, 24, 10)

	if err != nil {
		panic(err)
	}
	fmt.Println(n.String())

	n.TrainMnist()
	n.TestMnist()

}
