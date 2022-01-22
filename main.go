package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func main() {
	n, err := neuralnetwork.NewNetwork(neuralnetwork.DefaultConfig, 28*28, 28, 10)
	if err != nil {
		panic(err)
	}
	fmt.Println("Before training:")
	fmt.Println(n.String())
	//n.TrainMnist()
	data := make([]float64, 28*28)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float64()
	}

	n.ForwardFeed(data)
	fmt.Println("Forward Feed:")
	fmt.Println(n.String())

	expected := []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}
	_, prime := n.CostFunction.Cost(n.OutputLayer.Outputs, expected)
	n.Backpropagate(prime)

	fmt.Println("After training:")
	fmt.Println(n.String())
	//n.TestMnist()
}
