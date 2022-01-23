package neuralnetwork

import (
	"math"
)

//Activation will handle the different activation types

var activationTypes = []string{
	"sigmoid", "relu", "linear", "binary-step", "tanh", "arctan",
	"swish", "elu", "gelu", "selu",
}

type activation interface {
	fire(float64) float64
	derivative(float64) float64
}

type Sigmoid struct{}

//decided instead of 3 different Relu activations, we could
//just have one modular relu activation that covers any base
//want
type Relu struct {
	Leak     bool
	Cap      bool
	LeakAmnt float64
	CapAmnt  float64
}

func relu(leak ...float64) *Relu {
	r := Relu{}
	if len(leak) > 0 {
		r.Leak = true
		r.LeakAmnt = leak[0]
		if len(leak) > 1 && leak[1] > 1 {
			r.Cap = true
			r.CapAmnt = leak[1]
		}
	}
	return &r
}

type Linear struct{}

type BinaryStep struct{}

type Tanh struct{}

type ArcTan struct{}

type Swish struct {
	Sigmoid *Sigmoid
}

func getActivation(activation int, leak ...float64) activation {
	switch activation {
	case 0:
		return &Sigmoid{}
	case 1:
		return relu(leak...)
	case 2:
		return &Linear{}
	case 3:
		return &BinaryStep{}
	case 4:
		return &Tanh{}
	case 5:
		return &ArcTan{}
	case 6:
		return &Swish{
			Sigmoid: &Sigmoid{},
		}
	default:
		return relu(leak...)
	}
}

//Sigmoid takes an input a and applies the logistic sigmoid
//function, returning an activation between 0 and 1
func (s *Sigmoid) fire(a float64) float64 {
	return (1 / (math.Exp(-a) + 1))
}

//Relu is a nonlinear activation function that takes an input
//a, and returns a if the input is greater than zero. Otherwise
//the function returns 0
func (r *Relu) fire(a float64) float64 {
	if a < 0 {
		if r.Leak {
			return r.LeakAmnt
		}
		return 0
	}
	if r.Cap {
		if a > r.CapAmnt {
			return r.CapAmnt
		}
	}
	return a
}

//Linear uses a linear activation function
func (l *Linear) fire(a float64) float64 {
	return a
}

//Binary step takes an input a and returns 1 if a is greater
//than zero. Otherwise the function returns 0
func (b *BinaryStep) fire(a float64) float64 {
	if a < 0 {
		return 0
	}
	return 1
}

//Tanh takes an input, a , and maps it to the hyperbolic
//tangent function, producing a probability in the range
//-1, 1
func (t *Tanh) fire(a float64) float64 {
	return math.Tanh(a)
}

//ArcTan takes an input a, and returns the arctangent
//value
func (arc *ArcTan) fire(a float64) float64 {
	return 1 / (1 + math.Exp(-a)) //doesn't make sense to me either
}

//Swish is weird, but probably really good?
func (s *Swish) fire(a float64) float64 {
	//this needs updated, it's an incomplete swish
	return a * s.Sigmoid.fire(a)
}

/*
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
*/
