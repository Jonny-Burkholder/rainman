package neuralnetwork

import "math"

//Activation will handle the different activation types

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
		return .001 * a
	}
	return a
}

//LeakyRelu6 combines leaky relu with a max cap
//of 6. Eventually I'd like to figure out how to
//make the cap variable based on either the network's
//config, or some parameter in the layer
func LeakyRelu6(a float64) float64 {
	if a < 0 {
		return .001 * a
	} else if a > 6 {
		return 6
	} else {
		return a
	}
}

//Swish is weird, but probably really good?
func Swish(a float64) float64 {
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
