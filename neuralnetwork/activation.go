package neuralnetwork

import (
	"math"
)

//Activation will handle the different activation types

var activationTypes = []string{
	"sigmoid", "relu", "linear", "binary-step", "tanh", "arctan",
	"leakyrelu", "leakyrelu6", "swish", "elu", "gelu", "selu",
}

func getActivation(activation int) func(float64) float64 {
	switch activation {
	case 0:
		return Sigmoid
	case 1:
		return Relu
	case 2:
		return Linear
	case 3:
		return BinaryStep
	case 4:
		return Tanh
	case 5:
		return ArcTan
	case 6:
		return LeakyRelu
	case 7:
		return LeakyRelu6
	default:
		return LeakyRelu6
	}
}

//Sigmoid takes an input a and applies the logistic sigmoid
//function, returning an activation between 0 and 1
func Sigmoid(a float64) float64 {
	return math.Exp(a) / (math.Exp(a) + 1)
}

//Relu is a nonlinear activation function that takes an input
//a, and returns a if the input is greater than zero. Otherwise
//the function returns 0
func Relu(a float64) float64 {
	if a < 0 {
		return 0
	}
	return a
}

//Linear uses a linear activation function
func Linear(a float64) float64 {
	return a
}

//Binary step takes an input a and returns 1 if a is greater
//than zero. Otherwise the function returns 0
func BinaryStep(a float64) float64 {
	if a < 0 {
		return 0
	}
	return 1
}

//Tanh takes an input, a , and maps it to the hyperbolic
//tangent function, producing a probability in the range
//-1, 1
func Tanh(a float64) float64 {
	return math.Tanh(a)
}

//ArcTan takes an input a, and returns the arctangent
//value
func ArcTan(a float64) float64 {
	return 1 / (1 + math.Exp(-a)) //doesn't make sense to me either
}

//LeakyRelu takes an input a, and returns a if a
//is greater than zero. Otherwise, the function
//returns 1/1000 the input value
func LeakyRelu(a float64) float64 {
	if a < 0 {
		return .01 * a
	}
	return a
}

//LeakyRelu6 combines leaky relu with a max cap
//of 6. Eventually I'd like to figure out how to
//make the cap variable based on either the network's
//config, or some parameter in the layer
func LeakyRelu6(a float64) float64 {
	if a < 0 {
		return .01 * a
	} else if a > 6 {
		return 6
	} else {
		return a
	}
}

//Swish is weird, but probably really good?
func Swish(a float64) float64 {
	//This I guess should eventually involve a constant,
	//probably at the neuron level. I can make that a
	//slice of float64 on the layer level, I think
	return a * Sigmoid(a)
}

//Elu
func Elu(a float64) float64 {
	return a
}

//Gelu
func Gelu(a float64) float64 {
	return a
}

//Selu
func Selu(a float64) float64 {
	return a
}
