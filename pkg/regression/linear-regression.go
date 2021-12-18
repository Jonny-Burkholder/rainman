package regression

import "math"

//For now, a line only deals with 2 dimensions, because my poor brain
type Line struct {
	//Dimensions int //we'll add this later
	Slope     float32 //No idea how this works
	Intercept float32
}

//NewLine returns an empty line
func NewLine() *Line {
	return &Line{}
}

//Fit fits a line to x and y data. For now it just reads the whole
//thing, it doesn't randomize or cap the input at all
func (l *Line) Fit(y, x []float32) {

	var yavg, xavg float32 = 0, 0

	for i := 0; i < len(y); i++ {
		yavg += y[i]
	}

	yavg /= float32(len(y))

	for i := 0; i < len(x); i++ {
		xavg += x[i]
	}

	xavg /= float32(len(x))

	var num, den float32

	//this loop should break if we go out of bounds
	for i := 0; i != len(y) || i != len(x); i++ {
		a := x[i] - xavg
		b := y[i] - yavg

		num += a * b
		//may have to change to float64 to avoid all this casting
		den += float32(math.Pow(float64(a), 2))
	}

	l.Slope = num / den

	l.Intercept = yavg - (l.Slope * xavg)

}

//Predict does what it says on the tin
func (l *Line) Predict(v float32) float32 {
	return l.Intercept + (l.Slope * v)
}
