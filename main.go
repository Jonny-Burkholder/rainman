package main

import (
	"fmt"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func main() {
	n, err := neuralnetwork.NewNetwork(neuralnetwork.NewDefaultConfig(), 28*28, 28, 10)
	if err != nil {
		panic(err)
	}
	fmt.Println("Before training:")
	fmt.Println(n.String())
	n.TrainMnist()
	fmt.Println("After training:")
	fmt.Println(n.String())
	n.TestMnist()
}
