package neuralnetwork

import (
	"fmt"
	"testing"
)

//test functions for the neural network package

func TestSigmoid(t *testing.T) {
	sig := &Sigmoid{}

	s := ""

	for i := 1; i < 100; i += 10 {
		s += fmt.Sprint(sig.fire(float64(i)))
		s += "\n"
	}

	if 1 != 2 {
		t.Errorf("%s", s)
	}

}
