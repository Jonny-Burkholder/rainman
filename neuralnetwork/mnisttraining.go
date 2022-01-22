package neuralnetwork

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Jonny-Burkholder/neural-network/mnist"
)

//TrainMnist specifically just trains the network to read mnist data
func (n *Network) TrainMnist() {
	data, err := mnist.ReadTrainSet("./mnist/dataset")
	if err != nil {
		panic(err)
	}

	//seed random so that we actually get random numbers
	rand.Seed(time.Now().UnixNano())

	//let's shuffle our index numbers
	randomIndex := make([]int, data.N)
	for i, num := range rand.Perm(data.N) {
		randomIndex[i] = num
	}

	//let's set some variables
	var avgErr float64 = 1000
	i := 0
	chunkSize := n.Config.StochasticMax

	//run until either we run out of data, or the config file tells us to stop
	for i*chunkSize < data.N && i < n.Config.MaxSteps && avgErr > n.Config.TrainingCondition {
		//let's grab the index numbers for this chunk
		startIndex := chunkSize * i
		endIndex := startIndex + chunkSize
		chunk := randomIndex[startIndex:endIndex]

		//let's train through a chunk of data using that random index slice
		for _, j := range chunk {
			//first, we have to convert the mnist data into something we can read
			input := unpackExample(data.Data[j].Image)
			out := n.ForwardFeed(input)
			expected := make([]float64, len(out))
			expected[data.Data[j].Digit] = 1.0
			cost, prime := n.CostFunction.Cost(out, expected)
			n.Backpropagate(prime)
			avgErr += cost
		}

		//it is *average* error, after all
		avgErr /= float64(chunkSize)

	}

}

//this is going to be expensive. Whoops
//Adding sigmoid to each input to further PUNISH my cpu.
//and also because all of my weights are getting turned
//to NaN, which I don't like, and might be becuase of the
//massive values I'm putting into the system. I think I'm
//getting overflow somewhere. Poor rainman's brain is exploding!
//too much input.
//Ok turns out that wasn't the problem, and I might have something
//where I'm trying to calculate an imaginary number somewhere
func unpackExample(e [][]uint8) []float64 {
	res := make([]float64, len(e)*len(e[0]))

	//sig := Sigmoid{}

	k := 0

	for i := 0; i < len(e); i++ {
		for j := 0; j < len(e[0]); j++ {
			//back to the old sigmoid ploy, eh?
			sig := Sigmoid{}
			//have to cap values at 100 for the sigmoid to work, because of how golang's
			//math.Exp() function works. Honestly... I'm not even sure why the mnist values
			//go so high. I'm sure I could figure out a useful way to squishify them between
			//0 and 100, but... what's the point lol
			if e[i][j] < 100 {
				res[k] = sig.fire(float64(e[i][j])) //there's probably a way bitshift this, instead of... this
			} else {
				res[k] = 100
			}
			k++
		}
	}
	return res
}

//decide is a helper function that picks the highest output of the network
func decide(res []float64) (int, float64) {
	h := 0
	var c float64
	for i, num := range res {
		if num > c {
			h = i
			c = num
		}
	}
	return h, c
}

//TestMnist will test a fully-trained network to see how much of the test
//data it can correctly recognize
func (n *Network) TestMnist() {
	var res string
	var correct, incorrect int
	var avgCertainty float64
	//do stuff and print stuff here
	start := time.Now()
	data, err := mnist.ReadTestSet("./mnist/dataset")
	if err != nil {
		panic(err)
	}
	for i := 0; i < data.N; i++ {
		input := unpackExample(data.Data[i].Image)
		num, certainty := decide(n.ForwardFeed(input))
		if num != data.Data[i].Digit {
			incorrect++
		} else {
			correct++
		}
		avgCertainty += certainty
	}
	//good news is, there's no need to randomize anything
	//don't even need to return, we'll just track everything
	//and print from here
	duration := time.Now().Sub(start)

	res += fmt.Sprintf("Tested %v examples in %v seconds\n\n", data.N, duration.Seconds())
	res += fmt.Sprintf("Tested Correct: %v\n", correct)
	res += fmt.Sprintf("Tested Incorrect: %v\n", incorrect)
	res += fmt.Sprintf("Percent Correct: %2.2f\n", float64(correct)/float64(data.N)*100)
	res += fmt.Sprintf("Average Certainty: %v\n", avgCertainty)

	fmt.Print(res)
}
