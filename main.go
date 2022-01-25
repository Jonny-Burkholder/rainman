package main

import (
	"fmt"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func main() {
	n, err := neuralnetwork.NewNetwork(neuralnetwork.DefaultConfig, 28*28, 48, 28, 10)
	//n, err := neuralnetwork.NewNetwork(neuralnetwork.DefaultConfig, 10, 8, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println("Before training:")
	fmt.Println(n.String())

	/*
		rand.Seed(time.Now().UnixNano())
		data := make([]float64, 10)
		for j := 0; j < 10; j++ {
			data[j] = rand.Float64()
		}
		expected := []float64{0, 0, 0, 1}

		for i := 0; i < 21; i++ {
			n.ForwardFeed(data)
			cost, prime := n.CostFunction.Cost(n.OutputLayer.Outputs, expected)
			if i%10 == 0 {
				fmt.Printf("Iteration %v:\n", i)
				fmt.Println(cost)
				fmt.Println(averageCost(cost))
				fmt.Println(n.OutputLayer.Biases)
				fmt.Println(n.String())
			}

			n.Backpropagate(prime)
		}
	*/

	n.TrainMnist()
	fmt.Println("After training:")
	fmt.Println(n.String())
	n.TestMnist()

	/*
		nums := make([]float64, 28*28)
		rand.Seed(time.Now().UnixNano())
		for i := 0; i < 10; i++ {
			out := ""
			for j := 0; j < 28*28; j++ {
				nums[j] = rand.Float64()
			}
			res := n.ForwardFeed(nums)
			for _, num := range res {
				out += fmt.Sprintf("%v ", num)
			}
			fmt.Println("Output:")
			fmt.Println(out)
		}
		fmt.Println(n.String())
	*/
}

func averageCost(val []float64) float64 {
	var res float64
	for i := 0; i < len(val); i++ {
		res += val[i]
	}
	return res / float64(len(val))
}
