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

	rand.Seed(time.Now().UnixNano())
	data := make([]float64, 10)
	expected := []float64{0, 0, 0, 0, 4, 0, 0, 0, 0, 0}

	for i := 0; i < 5; i++ {
		for j := 0; j < 10; j++ {
			data[j] = rand.Float64()
		}
		n.ForwardFeed(data)
		_, prime := n.CostFunction.Cost(n.OutputLayer.Outputs, expected)
		n.Backpropagate(prime)
	}
	//n.TrainMnist()
	fmt.Println("After training:")
	fmt.Println(n.String())
	//n.TestMnist()
}
