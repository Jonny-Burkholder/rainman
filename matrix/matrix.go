package rainman

import "fmt"

type DataType = float64

type Matrix struct {
	Rows, Cols int
	Data       []DataType
	Stride     int
}

func NewMatrix(rows, cols int, data []DataType) *Matrix {
	if rows < 0 || cols < 0 {
		panic("rows and columns must be greater than zero")
	}
	if data == nil {
		data = make([]DataType, rows*cols)
	}
	return &Matrix{
		Rows:   rows,
		Cols:   cols,
		Data:   data,
		Stride: cols,
	}
}

func newEmptyMatrix() *Matrix {
	return &Matrix{
		Rows:   0,
		Cols:   0,
		Data:   make([]DataType, 0),
		Stride: 0,
	}
}

func CopyOf(other *Matrix) *Matrix {
	nm := newEmptyMatrix()
	nm.CloneFrom(other)
	return nm
}

func (m *Matrix) Reset() {
	if m == nil {
		panic("nothing to reset, receiver is nil")
	}
	m.Rows = 0
	m.Cols = 0
	m.Data = m.Data[:0]
	m.Stride = 0
}

func (m *Matrix) IsEmpty() bool {
	if m == nil {
		return true
	}
	return m.Stride == 0
}

func (m *Matrix) Equal(other *Matrix) bool {
	if m == nil && other != nil || m != nil && other == nil {
		return false
	}
	if m.Rows != other.Rows {
		return false
	}
	if m.Cols != other.Cols {
		return false
	}
	if m.Stride != other.Stride {
		return false
	}
	if len(m.Data) != len(other.Data) {
		return false
	}
	if cap(m.Data) != cap(other.Data) {
		return false
	}
	for i := range m.Data {
		if m.Data[i] != other.Data[i] {
			return false
		}
	}
	return true
}

func (m *Matrix) CloneFrom(other *Matrix) {
	if m == nil {
		panic("cannot clone into a nil receiver")
	}
	m.Reset()
	if cap(m.Data) != cap(other.Data) {
		m.Data = make([]DataType, other.Rows*other.Cols)
	}
	m.Rows = other.Rows
	m.Cols = other.Cols
	copy(m.Data, other.Data)
	m.Stride = other.Stride
}

func (m *Matrix) GetRow(n int) []DataType {
	if n >= m.Rows || n < 0 {
		panic("invalid row access attempt")
	}
	row := make([]DataType, m.Cols)
	i := m.Cols * n
	copy(row, m.Data[i:i+m.Cols])
	return row
}

func (m *Matrix) GetCol(n int) []DataType {
	if n >= m.Cols || n < 0 {
		panic("invalid column access attempt")
	}
	var col []DataType
	for i := 0; i < m.Rows; i++ {
		col = append(col, m.GetRow(i)[n])
	}
	return col
}

func (m *Matrix) GetAt(r, c int) DataType {
	return m.GetRow(r)[c]
}

func (m *Matrix) SetRow(n int, src []DataType) {
	if n >= m.Rows || n < 0 {
		panic("invalid row access attempt")
	}
	if len(src) != m.Cols {
		panic("invalid row length")
	}
	i := m.Cols * n
	copy(m.Data[i:i+m.Cols], src)
}

func (m *Matrix) SetCol(n int, src []DataType) {
	if n >= m.Cols || n < 0 {
		panic("invalid column access attempt")
	}
	if len(src) != m.Rows {
		panic("invalid column length")
	}
	for i := 0; i < m.Rows; i++ {
		m.SetAt(i, n, src[i])
	}
}

func (m *Matrix) SetAt(r, c int, v DataType) {
	if r >= m.Rows || r < 0 {
		panic("invalid row access attempt")
	}
	if c >= m.Cols || c < 0 {
		panic("invalid column access attempt")
	}
	i := m.Cols * r
	m.Data[i : i+m.Cols][c] = v
}

// Apply runs function fn on a copy the elements of m. Matrix m remains unchanged
// but the new copy of the transformed Matrix is returned
func (m *Matrix) Apply(fn func(r, c int, cell DataType) DataType) *Matrix {
	nm := CopyOf(m)
	for mr := 0; mr < nm.Rows; mr++ {
		for mc := 0; mc < nm.Cols; mc++ {
			i := nm.Cols * mr
			cell := nm.Data[i : i+nm.Cols][mc]
			nm.Data[i : i+nm.Cols][mc] = fn(mr, mc, cell)
		}
	}
	return nm
}

// Transform runs function fn on the elements of m. It transforms in place.
func (m *Matrix) Transform(fn func(r, c int, cell DataType) DataType) {
	for mr := 0; mr < m.Rows; mr++ {
		for mc := 0; mc < m.Cols; mc++ {
			i := m.Cols * mr
			cell := m.Data[i : i+m.Cols][mc]
			m.Data[i : i+m.Cols][mc] = fn(mr, mc, cell)
		}
	}
}

// Scale multiplies the elements of m by f. It transforms in place.
func (m *Matrix) Scale(f float64) *Matrix {
	m.quickTransform(func(r, c int, cell DataType) DataType {
		return f * cell
	})
	return m
}

func (m *Matrix) Scalar(x DataType) *Matrix {

	return m
}

func Dot(a, b *Matrix) *Matrix {
	if a == nil || b == nil {
		panic("got nil receiver!")
	}
	var t DataType
	c := NewMatrix(a.Rows, b.Cols, nil)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			for k := 0; k < b.Rows; k++ {
				//fmt.Printf("a[%d][%d]=%v, b[%d][%d]=%v\n", i, k, a.GetAt(i, k), k, j, b.GetAt(k, j))
				t += a.GetAt(i, k) * b.GetAt(k, j)
			}
			c.SetAt(i, j, t)
			t = 0
		}
	}
	return c
}

func Add(a, b *Matrix) *Matrix {
	if a == nil || b == nil {
		panic("got nil receiver!")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrices must have the same dimensions in order to sum")
	}
	c := NewMatrix(a.Rows, b.Cols, nil)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			//c.Data[i+c.Stride*j] = a.Data[i+a.Stride*j] + b.Data[i+b.Stride*j]
			c.SetAt(i, j, a.GetAt(i, j)+b.GetAt(i, j))
		}
	}
	return c
}

func Sub(a, b *Matrix) *Matrix {
	if a == nil || b == nil {
		panic("got nil receiver!")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrices must have the same dimensions in order to sum")
	}
	c := NewMatrix(a.Rows, b.Cols, nil)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			c.SetAt(i, j, a.GetAt(i, j)-b.GetAt(i, j))
			//c.Data[i+c.Stride*j] = a.Data[i+a.Stride*j] - b.Data[i+b.Stride*j]
		}
	}
	return c
}

func Mul(a, b *Matrix) *Matrix {
	if a == nil || b == nil {
		panic("got nil receiver!")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrices must have the same dimensions in order to simple mul, otherwise use dot")
	}
	var t DataType
	c := NewMatrix(a.Rows, b.Cols, nil)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			for k := 0; k < b.Rows; k++ {
				//fmt.Printf("a[%d][%d]=%v, b[%d][%d]=%v\n", i, k, a.GetAt(i, k), k, j, b.GetAt(k, j))
				t += a.GetAt(i, k) * b.GetAt(k, j)
			}
			c.SetAt(i, j, t)
			t = 0
		}
	}
	return c
}

func Inv(m, m2 *Matrix) *Matrix {
	// TODO: need to look up the formula again
	return m
}

func (m *Matrix) quickSet(r, c int, v DataType) {
	if r < 0 || m.Rows <= r || c < 0 || m.Cols <= c {
		panic(fmt.Sprintf("out of bounds set(%d, %d): bounds r=%d, c=%d", r, c, m.Rows, m.Cols))
	}
	m.Data[r+m.Stride*c] = v
}

func (m *Matrix) quickGet(r, c int) DataType {
	if r < 0 || m.Rows <= r || c < 0 || m.Cols <= c {
		panic(fmt.Sprintf("out of bounds set(%d, %d): bounds r=%d, c=%d", r, c, m.Rows, m.Cols))
	}
	v := m.Data[r+m.Stride*c]
	return v
}

func (m *Matrix) quickTransform(fn func(r, c int, cell DataType) DataType) {
	for mrow := 0; mrow < m.Rows; mrow++ {
		for mcol := 0; mcol < m.Cols; mcol++ {
			oval := m.Data[mrow+m.Stride*mcol]
			nval := fn(mrow, mcol, oval)
			m.Data[mrow+m.Stride*mcol] = nval
		}
	}
}
