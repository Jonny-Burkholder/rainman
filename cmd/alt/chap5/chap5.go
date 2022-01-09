package main

import (
	"fmt"
	"github.com/Jonny-Burkholder/neural-network/pkg/matlib"
	"math"
)

// x is standard notation for input data
var x = [][]float64{
	{1.0, 2.0, 3.0, 2.5},
	{2.0, 5.0, -1.0, 2.0},
	{-1.5, 2.7, 3.3, -0.8},
}

var inputs = []float64{0.0, 2.0, -1.0, 3.3, -2.7, 1.1, 2.2, -100}
var output = []float64{}

func ReLU(input, output []float64) {
	for i := range input {
		output = append(output, math.Max(0, input[i]))
	}
}

func main() {
	l1 := NewLayerDense(4, 5)
	l2 := NewLayerDense(5, 2)

	l1.forward(x)
	fmt.Println(matlib.MatrixToString(l1.output))
	l2.forward(l1.output)
	fmt.Println(matlib.MatrixToString(l2.output))
}

type LayerDense struct {
	weights [][]float64
	biases  []float64
	output  [][]float64
}

func NewLayerDense(inputs, neurons int) *LayerDense {
	weights := matlib.NewMatrix(inputs, neurons)
	weights = matlib.MatrixFillRand(weights, nil)
	weights = matlib.MatrixApply(weights, func(x float64) float64 {
		return 0.1 * x
	})
	return &LayerDense{
		weights: weights,
		biases:  make([]float64, neurons),
	}
}

func (l *LayerDense) forward(inputs [][]float64) {
	l.output = matlib.MatrixDotProduct(inputs, l.weights)
	for i := range l.output {
		for j := range l.output[i] {
			l.output[i][j] += l.biases[j]
		}
	}
}

func (l *LayerDense) String() string {
	ss := fmt.Sprintf("layer.weights=")
	ss += fmt.Sprintf(matlib.MatrixToString(l.weights))
	return ss
}
