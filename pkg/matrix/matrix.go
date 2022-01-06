package matrix

import (
	"fmt"
	"unsafe"
)

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

func (m *Matrix) IsEqual(other *Matrix) bool {
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
	row := make([]DataType, m.Stride)
	i := m.Stride * n
	copy(row, m.Data[i:i+m.Stride])
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
	//return m.GetRow(r)[c]
	return m.quickGet(r, c)
}

func (m *Matrix) SetRow(n int, src []DataType) {
	if n >= m.Rows || n < 0 {
		panic("invalid row access attempt")
	}
	if len(src) != m.Cols {
		panic("invalid row length")
	}
	i := m.Stride * n
	copy(m.Data[i:i+m.Stride], src)
}

func (m *Matrix) SetCol(n int, src []DataType) {
	if n >= m.Cols || n < 0 {
		panic("invalid column access attempt")
	}
	if len(src) != m.Rows {
		panic("invalid column length")
	}
	for i := 0; i < m.Rows; i++ {
		//m.SetAt(i, n, src[i])
		m.quickSet(i, n, src[i])
	}
}

func (m *Matrix) SetAt(r, c int, v DataType) {
	if r >= m.Rows || r < 0 {
		panic("invalid row access attempt")
	}
	if c >= m.Cols || c < 0 {
		panic("invalid column access attempt")
	}
	m.quickSet(r, c, v)
	//i := m.Cols * r
	//m.Data[i : i+m.Cols][c] = v
}

// Apply runs function fn on a copy the elements of m. Matrix m remains unchanged
// but the new copy of the transformed Matrix is returned
func (m *Matrix) Apply(fn func(r, c int, cell DataType) DataType) *Matrix {
	nm := CopyOf(m)
	for mr := 0; mr < nm.Rows; mr++ {
		for mc := 0; mc < nm.Cols; mc++ {
			cell := m.quickGet(mr, mc)
			m.quickSet(mr, mc, fn(mr, mc, cell))
		}
	}
	return nm
}

// Transform runs function fn on the elements of m. It transforms in place.
func (m *Matrix) Transform(fn func(r, c int, cell DataType) DataType) {
	for mr := 0; mr < m.Rows; mr++ {
		for mc := 0; mc < m.Cols; mc++ {
			cell := m.quickGet(mr, mc)
			m.quickSet(mr, mc, fn(mr, mc, cell))
		}
	}
}

// Scale multiplies the elements of m by f. It transforms in place.
func (m *Matrix) Scale(f float64) *Matrix {
	m.Transform(func(r, c int, cell DataType) DataType {
		return f * cell
	})
	return m
}

func (m *Matrix) Scalar(x DataType) *Matrix {
	// TODO: implement or remove
	return m
}

func CopyOf(other *Matrix) *Matrix {
	nm := newEmptyMatrix()
	nm.CloneFrom(other)
	return nm
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
				t += a.quickGet(i, k) * b.quickGet(k, j)
			}
			c.quickSet(i, j, t)
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
			c.quickAdd(i, j, a, b)
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
			c.quickSub(i, j, a, b)
		}
	}
	return c
}

func Inv(a, b *Matrix) *Matrix {
	// TODO: need to look up the formula again
	c := NewMatrix(a.Rows, b.Cols, nil)
	_ = c // remove after implement
	return c
}

func (m *Matrix) quickSet(r, c int, v DataType) {
	if r < 0 || m.Rows <= r || c < 0 || m.Cols <= c {
		panic(fmt.Sprintf("out of bounds set(%d, %d): bounds r=%d, c=%d", r, c, m.Rows, m.Cols))
	}
	m.Data[r*m.Stride+c] = v
}

func (m *Matrix) quickGet(r, c int) DataType {
	if r < 0 || m.Rows <= r || c < 0 || m.Cols <= c {
		panic(fmt.Sprintf("out of bounds set(%d, %d): bounds r=%d, c=%d", r, c, m.Rows, m.Cols))
	}
	v := m.Data[r*m.Stride+c]
	return v
}

func (m *Matrix) quickAdd(i, j int, a, b *Matrix) {
	m.quickSet(i, j, a.quickGet(i, j)+b.quickGet(i, j))
}

func (m *Matrix) quickSub(i, j int, a, b *Matrix) {
	m.quickSet(i, j, a.quickGet(i, j)-b.quickGet(i, j))
}

func (m *Matrix) Length() int {
	return m.Rows * m.Cols
}

func (m *Matrix) Size() int {
	var size int
	size += int(unsafe.Sizeof(m.Data)) + (cap(m.Data) * int(unsafe.Sizeof(DataType(0))))
	size += int(unsafe.Sizeof(m.Rows))
	size += int(unsafe.Sizeof(m.Cols))
	size += int(unsafe.Sizeof(m.Stride))
	return size
}

func (m *Matrix) String() string {
	ss := fmt.Sprintf("{")
	for mr := 0; mr < m.Rows; mr++ {
		ss += fmt.Sprintf("{")
		for mc := 0; mc < m.Cols; mc++ {
			ss += fmt.Sprintf("%v", m.quickGet(mr, mc))
			if mc < m.Cols-1 {
				ss += fmt.Sprintf(",")
			}
		}
		ss += fmt.Sprintf("}")
		if mr < m.Rows-1 {
			ss += fmt.Sprintf(",\n ")
		}
	}
	ss += fmt.Sprintf("}\n")
	return ss
}
