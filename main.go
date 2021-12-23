package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func main() {

	config := neuralnetwork.NewDefaultConfig()

	nn, err := neuralnetwork.NewNetwork("test_network", config, 16, 8, 4, 2)

	fmt.Println(nn.String())

	if err != nil {
		panic(err)
	}

	rand.Seed(time.Now().UnixNano())

	rand.Seed(rand.Int63())

	data := make([]float64, 16)

	for i := 0; i < 16; i++ {
		data[i] = rand.Float64()
	}

	fmt.Println("Data:")
	fmt.Println(data)
	fmt.Println("Output:")

	output, err := nn.Activate(data)

	if err != nil {
		panic(err)
	}

	fmt.Println(output)

}
