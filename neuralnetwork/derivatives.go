package neuralnetwork

//SigmoidPrime returns the derivative for the
//logistic sigmoid activation function
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

//ReluPrime returns the derivative for the
//relu activation function. It's kind of
//lame, to be honest
func ReluPrime(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

//LinearPrime is probably unnecessary, yeah?

//BinaryStepPrime returns the derivative of
//the binary step activation function
func BinaryStepPrime(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

//TanhPrime returns the derivative of the
//hyperbolic tangent activation function
func TanhPrime(x float64) float64 {
	res := Tanh(x)
	res *= res //should be significantly faster than doing the tanh function twice
	return 1 - res
}

//ArcTanPrime returns the derivative of the
//ArcTan activation function
func ArcTanPrime() {}

//LeakyReluPrime returns the derivative of
//the leaky relu activation function
func LeakyReluPrime(x, leak float64) float64 {
	if x > 0 {
		return 1
	} else {
		return leak
	}
}

//LeakyRelu6Prime returns the derivative of the
//leaky relu6 activation function, which just so
//happens to be the exact same derivative as the
//leaky relu function
func LeakyRelu6Prime(x, leak float64) float64 {
	if x > 0 {
		return 1
	} else {
		return leak
	}
}

//SwishPrime returns the derivative of the
//swish activation function
func SwishPrime(x float64) float64 {
	//eventually needs to include a constant!
	//Also this function feels like it would get bogged
	//down pretty quickly as you add neurons
	return Swish(x) + (Sigmoid(x) * (1 - Swish(x)))
}
