package main

import (
	"fmt"
	"github.com/Jonny-Burkholder/neural-network/pkg/ann"
	"time"
)

var samples = [][][]float64{
	{{0, 0}, {0}},
	{{0, 1}, {1}},
	{{1, 0}, {1}},
	{{1, 1}, {0}},
}

func main() {

	// testing ann out with the classic xor example.
	// in case you are not familiar will demonstrate
	// the xor operation below, so you can know what you
	// are looking at. xor is a binary operation kind
	// of like a transistor. basically if x, and y are
	// the same, the result of the xor operation is 0
	// and if x and y are different the result is a 1.

	// we make a simple xor demo func
	xor := func(x, y int) {
		res := x ^ y // <- note: ^ is the xor binary operation
		fmt.Printf("x=%d, y=%d, res=%d\n", x, y, res)
	}

	// and we try the same digits that are in our sample set
	xor(0, 0) // expecting 0
	xor(0, 1) // expecting 1
	xor(1, 0) // expecting 1
	xor(1, 1) // expecting 0

	// so now that we understand how that works, let's try
	// to NOT actually do the XOR operation, but have our
	// artificial neural network learn the algorithm and
	// apply it for us. do it!

	// first lets set up the config to have 3 layers.
	// an input layer with 2 neurons, a hidden layer
	// with three neurons, and an output layer with
	// 1 neuron.
	conf := ann.Config{
		Layers:       []int{2, 3, 1},
		Seed:         time.Now().UnixNano(),
		IsRegression: false,
		Activation:   ann.ReLU,
	}

	// set up the network...
	nn := ann.NewANN(&conf)
	// let's print out the weights in the network
	fmt.Printf("%s\n", nn)

	// ...and train with our sample set above.
	// we will do one hundred thousand iterations
	// with a learn rate of 1.001 and no debug
	fmt.Printf("Start training... ")
	t := time.Now()
	nn.Train(samples, 100000, 1.001, false)
	fmt.Printf("finished training (took %v)\n", time.Now().Sub(t))

	fmt.Println("Running predictions...")
	fmt.Println(nn.Predict([]float64{0, 0}))
	fmt.Println(nn.Predict([]float64{0, 1}))
	fmt.Println(nn.Predict([]float64{1, 0}))
	fmt.Println(nn.Predict([]float64{1, 1}))
	fmt.Println("done.")
}
