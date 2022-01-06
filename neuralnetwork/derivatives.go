package neuralnetwork

/*These were previously methods on neurons, but
that was stupid so I changed it*/

//SigmoidPrime returns the derivative for the
//logistic sigmoid activation function
func (s *Sigmoid) Derivative(x float64) float64 {
	return s.fire(x) * (1 - s.fire(x))
}

//ReluPrime returns the derivative for the
//relu activation function. It's kind of
//lame, to be honest
func (r *Relu) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	} else if r.Leak {
		return r.LeakAmnt
	} else {
		return 0
	}
}

//LinearPrime is probably unnecessary, yeah?

//BinaryStepPrime returns the derivative of
//the binary step activation function
func (b *BinaryStep) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

//TanhPrime returns the derivative of the
//hyperbolic tangent activation function
func (t *Tanh) Derivative(x float64) float64 {
	res := t.fire(x)
	res *= res //should be significantly faster than doing the tanh function twice
	return 1 - res
}

//ArcTanPrime returns the derivative of the
//ArcTan activation ft.fireion
func (arc *ArcTan) Derivative() {}

//SwishPrime returns the derivative of the
//swish activation function
func (s *Swish) Derivative(x float64) float64 {
	//eventually needs to include a constant!
	//Also this function feels like it would get bogged
	//down pretty quickly as you add neurons
	return s.fire(x) + (s.Sigmoid.fire(x) * (1 - s.fire(x)))
}
