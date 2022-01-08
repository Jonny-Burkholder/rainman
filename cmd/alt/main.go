package main

import (
	"fmt"
	"github.com/Jonny-Burkholder/neural-network/pkg/matlib"
)

// Code from me following along and transpiling the python.
// The resources that I am following can be found below.
// -----------------------------------------------------
// Neural Networks from Scratch book: https://nnfs.io
// NNFSiX Github: https://github.com/Sentdex/NNfSiX

func main() {

}

func chapter3() {
	inputs := [][]float64{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	weights := [][]float64{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	}

	biases := []float64{2.0, 3.0, 0.5}
	var output [][]float64

	output = matlib.MatrixDotProduct(inputs, matlib.MatrixTranspose(weights))
	for i := range output {
		for j := range output[i] {
			output[i][j] += biases[j]
		}
	}
	fmt.Printf("%v", output)
}

type nnet struct {
	inputs  [][]float64
	weights [][]float64
	biases  []float64
	output  [][]float64
	actFn
}

type actFn func(x float64) float64

func newNNet(conf []int, act actFn) *nnet {
	return &nnet{
		inputs:  make([][]float64, 0),
		weights: make([][]float64, 0),
		biases:  make([]float64, 0),
		output:  make([][]float64, 0),
		actFn:   act,
	}
}

type vector []float64
type matrix [][]float64

func zip(a, b []float64) [][]float64 {
	var c [][]float64
	if len(a) < len(b) {
		c = make([][]float64, len(a))
	} else {
		c = make([][]float64, len(b))
	}
	for i := 0; i < len(c); i++ {
		c[i] = make([]float64, 2)
		c[i] = []float64{a[i], b[i]}
	}
	return c
}

func layerOutputs() []float64 {
	return nil
}
