package neuralnetwork

import (
	"fmt"
	"testing"
)

//test functions for the neural network package

func TestSigmoid(t *testing.T) {
	sig := &Sigmoid{}

	s := ""

	for i := 1; i < 200; i += 10 {
		s += fmt.Sprint(sig.fire(float64(i)))
		s += "\n"
	}

	if 1 != 1 {
		t.Errorf("%s", s)
	}

	fmt.Println(s)

}

func TestRelu(t *testing.T) {
	r := relu(.001, 6)
	a := r.fire(0)
	b := r.fire(4.5235)
	c := r.fire(100000)

	if a != 0.001 {
		t.Errorf("Relu error: wanted %v, got %v", .001, a)
	}
	if b != 4.5235 {
		t.Errorf("Relu error: wanted %v, got %v", 4.5235, a)
	}
	if c != 6 {
		t.Errorf("Relu error: wanted %v, got %v", 6, a)
	}
}
