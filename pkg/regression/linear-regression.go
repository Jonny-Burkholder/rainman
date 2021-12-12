package regression

import "math"

//For now, a line only deals with 2 dimensions, because my poor brain
type Line struct {
	//Dimensions int //we'll add this later
	Y        []float32
	X        []float32 //dependent variabls in n number of dimensios
	YAvg     float32
	XAvg     float32
	Slope    float32 //No idea how this works
	Inercept float32
}

//NewLine takes a series of y and x values and returns a new line
//Need to return an error, don't feel like it right now
func NewLine(y, x []float32) *Line {

	var yavg, xavg float32 = 0, 0

	for i := 0; i < len(y); i++ {
		yavg += y[i]
	}

	yavg /= float32(len(y))

	for i := 0; i < len(y); i++ {
		xavg += y[i]
	}

	xavg /= float32(len(x))

	return &Line{
		Y:    y,
		X:    x,
		YAvg: yavg,
		XAvg: xavg,
	}
}

//I forget what this is actually supposed to do
func (l *Line) Fit() {

	var num, den float32

	//this loop should break if we go out of bounds
	for i := 0; i != len(l.Y) || i != len(l.X); i++ {
		a := l.X[i] - l.XAvg
		b := l.Y[i] - l.YAvg

		num += a * b
		//may have to change to float64 to avoid all this casting
		den += float32(math.Pow(float64(a), 2))
	}

	l.Slope = num / den

	l.Inercept = l.YAvg - (l.Slope * l.XAvg)

}

//Predict does what it says on the tin
func (l *Line) Predict(v float32) float32 {
	return l.Inercept + (l.Slope * v)
}
