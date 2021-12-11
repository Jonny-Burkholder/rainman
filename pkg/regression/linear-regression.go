package regression

type Line struct {
	Dimensions int
	Points     []int
}

func NewLine(dimensions int, points ...int) *Line {
	return &Line{
		Dimensions: dimensions,
		Points:     points,
	}
}

//I forget what this is actually supposed to do
func (l *Line) Fit() {

}
