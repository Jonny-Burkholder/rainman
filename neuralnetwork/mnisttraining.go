package neuralnetwork

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Jonny-Burkholder/neural-network/mnist"
)

//TrainMnist specifically just trains the network to read mnist data
func (n *Network) TrainMnist() {

	start := time.Now()

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
	chunkSize := 100
	iter := 0

	//run until either we run out of data, or the config file tells us to stop
	//honestly not really sure why I'm even using chunk sizes here
	for i*chunkSize < data.N && i < n.Config.MaxSteps && avgErr > n.Config.TrainingCondition {
		//reset average error
		avgErr = 0

		//let's grab the index numbers for this chunk
		startIndex := chunkSize * i
		endIndex := startIndex + chunkSize
		chunk := randomIndex[startIndex:endIndex]

		//let's train through a chunk of data using that random index slice
		for _, j := range chunk {
			//just to make sure we aren't hanging anywhere
			if iter%1000 == 0 {
				fmt.Printf("Training %v examples...\n", iter)
			}
			//first, we have to convert the mnist data into something we can read
			input := unpackExample(data.Data[j].Image)
			out := n.ForwardFeed(input)
			expected := make([]float64, len(out))
			expected[data.Data[j].Digit] = 1.0
			cost, prime := n.CostFunction.Cost(out, expected)
			n.Backpropagate(prime)
			avgErr += averageCost(cost)
			//fmt.Printf("\nIteration %v\n", iter)
			//fmt.Println(n.String())
			iter++
		}

		//it is *average* error, after all
		avgErr /= float64(chunkSize)

		i++

	}

	var printout string

	printout += fmt.Sprintf("Trained %v examples in %1.2f seconds\n", iter, time.Now().Sub(start).Seconds())
	printout += fmt.Sprintf("Average Error: %1.4f", avgErr)

	fmt.Println(printout)

}

//compressCost averages a slice of cost
//values into a single value
func averageCost(cost []float64) float64 {
	var res float64
	for _, num := range cost {
		res += num
	}
	return res / float64(len(cost))
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
			//have to cap values at 100 for the sigmoid to work, because of how golang's
			//math.Exp() function works. Honestly... I'm not even sure why the mnist values
			//go so high. I'm sure I could figure out a useful way to squishify them between
			//0 and 100, but... what's the point lol
			res[k] = float64(e[i][j]) / 255
			k++
		}
	}
	return res
}

//decide is a helper function that picks the highest output of the network
func decide(res []float64) (int, []float64) {
	digit := 0
	var certainty = []float64{0, 0, 0}
	for i, num := range res {
		certainty[2] += num
		if num > certainty[0] {
			digit = i
			certainty[1] = num - certainty[0]
			certainty[0] = num
		}
	}
	certainty[2] /= float64(len(res))
	return digit, certainty
}

//TestMnist will test a fully-trained network to see how much of the test
//data it can correctly recognize
func (n *Network) TestMnist() {
	var printout string
	var correct, incorrect int
	avgCertainty := make([]float64, 3)
	//do stuff and print stuff here
	start := time.Now()
	data, err := mnist.ReadTestSet("./mnist/dataset")
	if err != nil {
		panic(err)
	}
	i := 0
	for i < data.N {
		if i%1000 == 0 {
			fmt.Printf("Testing %v examples...\n", i)
		}
		input := unpackExample(data.Data[i].Image)
		//fmt.Println("Input:")
		//fmt.Println(input)
		res := n.ForwardFeed(input)
		num, certainty := decide(res)
		if num != data.Data[i].Digit {
			incorrect++
		} else {
			correct++
		}

		for j := 0; j < len(avgCertainty); j++ {
			avgCertainty[j] += certainty[j]
		}

		if i%1000 == 0 {
			fmt.Printf("Predicted: %v\n", num)
			s := ""
			for j := 0; j < len(res); j++ {
				s += fmt.Sprintf("%1.4f ", res[j])
			}
			fmt.Println(s)
			fmt.Printf("Expected: %v\n", data.Data[i].Digit)
			mnist.PrintImage(data.Data[i].Image)
		}
		i++
	}

	for i := 0; i < len(avgCertainty); i++ {
		avgCertainty[i] /= float64(data.N)
	}

	//good news is, there's no need to randomize anything
	//don't even need to return, we'll just track everything
	//and print from here
	duration := time.Now().Sub(start)

	printout += fmt.Sprintf("Tested %v examples in %v seconds\n\n", data.N, duration.Seconds())
	printout += fmt.Sprintf("Tested Correct: %v\n", correct)
	printout += fmt.Sprintf("Tested Incorrect: %v\n", incorrect)
	printout += fmt.Sprintf("Percent Correct: %2.2f\n", float64(correct)/float64(data.N)*100)
	printout += fmt.Sprintf("Average Confidence: %1.2f\n", avgCertainty[0])
	printout += fmt.Sprintf("Average Certainty: %1.2f\n", avgCertainty[1])
	printout += fmt.Sprintf("Average Uncertainty: %1.2f\n", avgCertainty[2])

	fmt.Println(printout)
}
