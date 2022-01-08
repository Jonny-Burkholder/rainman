package nnv2

import (
	"math"
)

// more activation functions can be found below
// https://en.wikipedia.org/wiki/Activation_function

type ActivationFn interface {
	ActFn(x float64) float64
	DerFn(x float64) float64
}

func InitActivationFn(s string) ActivationFn {
	switch s {
	case "sigmoid":
		return Sigmoid
	case "tanh":
		return Tanh
	case "leakyReLU":
		return LeakyReLU
	case "reLU":
		return ReLU
	case "linear":
		return Linear
	}
	return defaultConfig.Activation
}

var (
	Sigmoid   = sigmoid{}
	Tanh      = tanh{}
	LeakyReLU = leakyReLU{}
	ReLU      = reLU{}
	Linear    = linear{}
)

type sigmoid struct{}

func (s sigmoid) ActFn(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

func (s sigmoid) DerFn(x float64) float64 {
	return x * (1 - x)
}

type tanh struct{}

func (t tanh) ActFn(x float64) float64 {
	return (1 - math.Exp(-2*x)) / (1 + math.Exp(-2*x))
}

func (t tanh) DerFn(y float64) float64 {
	return 1 - math.Pow(y, 2)
}

type leakyReLU struct{}

func (r leakyReLU) ActFn(x float64) float64 {
	if x < 0 {
		return 0.01 * x
	}
	return x
	//return math.Max(x, 0)
}

func (r leakyReLU) DerFn(y float64) float64 {
	if y < 0 {
		return 0.01
	}
	return 1
}

type reLU struct{}

func (r reLU) ActFn(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
	//return math.Max(x, 0)
}

func (r reLU) DerFn(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0
}

type linear struct{}

func (l linear) ActFn(x float64) float64 {
	return x
}

func (l linear) DerFn(x float64) float64 {
	return 1
}
