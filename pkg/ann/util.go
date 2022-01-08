package ann

import "fmt"

func sprint3DMatrix(m [][][]float64) string {
	var ss string
	for i := 0; i < len(m); i++ {
		ss += fmt.Sprintf("matrix[%d], ", i)
		ss += sprint2DMatrix(m[i])
	}
	return ss
}

func sprint2DMatrix(m [][]float64) string {
	ss := fmt.Sprintf("rows=%d, cols=%d\n", len(m), len(m[0]))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			ss += fmt.Sprintf("%.4f ", m[i][j])
		}
		ss += "\n"
	}
	ss += "\n"
	return ss
}
