package rainman

import (
	"fmt"
	"testing"
)

func TestNewMatrix(t *testing.T) {

	rows, cols := 3, 2

	m := NewMatrix(rows, cols, nil)
	if m == nil {
		t.Fail()
	}
	if m.Rows != rows {
		t.Fail()
	}
	if m.Cols != cols {
		t.Fail()
	}
	if len(m.Data) != rows*cols {
		t.Fail()
	}
}

func TestMatrix_Reset(t *testing.T) {

	rows, cols := 3, 2

	m := NewMatrix(rows, cols, nil)
	if m == nil {
		t.Fail()
	}
	if m.Rows != rows {
		t.Fail()
	}
	if m.Cols != cols {
		t.Fail()
	}
	if len(m.Data) != rows*cols {
		t.Fail()
	}

	m.Reset()

	if m.Rows != 0 {
		t.Fail()
	}
	if m.Cols != 0 {
		t.Fail()
	}
	if len(m.Data) != 0 {
		t.Fail()
	}
}

func TestMatrix_IsEmpty(t *testing.T) {

	rows, cols := 3, 2

	m := NewMatrix(rows, cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m.Reset()

	if !m.IsEmpty() {
		t.Fail()
	}
}

func TestMatrix_CloneFrom(t *testing.T) {

	rows, cols := 3, 2

	m := NewMatrix(rows, cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m2 := new(Matrix)

	if !m2.IsEmpty() {
		t.Fail()
	}

	m2.CloneFrom(m)

	if m2.IsEmpty() {
		t.Fail()
	}

	if !(m.Equal(m2) && m2.Equal(m)) {
		t.Fail()
	}

	m.SetAt(0, 1, 255.55)
	if m.GetAt(0, 1) != 255.55 {
		t.Fail()
	}

	if m2.GetAt(0, 1) == 255.55 {
		fmt.Println(">>>>>>>>>>>> DEBUG")
		t.Fail()
	}

}

type DataSet struct {
	Rows    int
	Cols    int
	Data    []DataType
	RowData [][]DataType
	ColData [][]DataType
}

func (ds DataSet) AtMatch(found DataType, r, c int) bool {
	if r > len(ds.RowData)-1 {
		return false
	}
	if c > len(ds.ColData)-1 {
		return false
	}
	match := ds.RowData[r][c]
	if found != match {
		return false
	}
	return true
}

func (ds DataSet) RowMatch(found []DataType, n int) bool {
	if n > len(ds.RowData)-1 {
		return false
	}
	match := ds.RowData[n]
	if len(found) != len(match) {
		return false
	}
	for i := range found {
		if found[i] != match[i] {
			return false
		}
	}
	return true
}

func (ds DataSet) ColMatch(found []DataType, n int) bool {
	if n > len(ds.ColData)-1 {
		return false
	}
	match := ds.ColData[n]
	if len(found) != len(match) {
		return false
	}
	for i := range found {
		if found[i] != match[i] {
			return false
		}
	}
	return true
}

var (
	dataSetA = DataSet{
		Rows:    3,
		Cols:    2,
		Data:    []DataType{2, 2, 0, 3, 0, 4},
		RowData: [][]DataType{{2, 2}, {0, 3}, {0, 4}},
		ColData: [][]DataType{{2, 0, 0}, {2, 3, 4}},
	}
	dataSetB = DataSet{
		Rows:    2,
		Cols:    3,
		Data:    []DataType{2, 1, 2, 3, 2, 4},
		RowData: [][]DataType{{2, 1, 2}, {3, 2, 4}},
		ColData: [][]DataType{{2, 3}, {1, 2}, {2, 4}},
	}
	dotProdSetASetBData = []DataType{10, 6, 12, 9, 6, 12, 12, 8, 16}

	addSubSetA = []DataType{2, 2, 0, 3, 0, 4}
	addSubSetB = []DataType{1, 2, 0, 1, 1, 3}

	addSetASetBData = []DataType{3, 4, 0, 4, 1, 7}
	subSetASetBData = []DataType{1, 0, 0, 2, -1, 1}
)

func TestMatrix_GetRow(t *testing.T) {

	dat := dataSetA

	m := NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if m.IsEmpty() {
		t.Fail()
	}

	r0 := m.GetRow(0)
	if !dat.RowMatch(r0, 0) {
		t.Fail()
	}

	r1 := m.GetRow(1)
	if !dat.RowMatch(r1, 1) {
		t.Fail()
	}

	r2 := m.GetRow(2)
	if !dat.RowMatch(r2, 2) {
		t.Fail()
	}

	m.Reset()

	dat = dataSetB

	m = NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if m.IsEmpty() {
		t.Fail()
	}

	r0 = m.GetRow(0)
	if !dat.RowMatch(r0, 0) {
		t.Fail()
	}

	r1 = m.GetRow(1)
	if !dat.RowMatch(r1, 1) {
		t.Fail()
	}

}

func TestMatrix_GetCol(t *testing.T) {

	dat := dataSetA

	m := NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if m.IsEmpty() {
		t.Fail()
	}

	c0 := m.GetCol(0)
	if !dat.ColMatch(c0, 0) {
		t.Fail()
	}

	c1 := m.GetCol(1)
	if !dat.ColMatch(c1, 1) {
		t.Fail()
	}

	m.Reset()

	dat = dataSetB

	m = NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if m.IsEmpty() {
		t.Fail()
	}

	c0 = m.GetCol(0)
	if !dat.ColMatch(c0, 0) {
		t.Fail()
	}

	c1 = m.GetCol(1)
	if !dat.ColMatch(c1, 1) {
		t.Fail()
	}

	c2 := m.GetCol(2)
	if !dat.ColMatch(c2, 2) {
		t.Fail()
	}

}

func TestMatrix_GetAt(t *testing.T) {

	dat := dataSetA

	m := NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if m.IsEmpty() {
		t.Fail()
	}

	a0 := m.GetAt(0, 1)
	if !dat.AtMatch(a0, 0, 1) {
		t.Fail()
	}

	a1 := m.GetAt(1, 1)
	if !dat.AtMatch(a1, 1, 1) {
		t.Fail()
	}

	a2 := m.GetAt(2, 1)
	if !dat.AtMatch(a2, 2, 1) {
		t.Fail()
	}

	m.Reset()

	dat = dataSetB

	m = NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if m.IsEmpty() {
		t.Fail()
	}

	a0 = m.GetAt(0, 1)
	if !dat.AtMatch(a0, 0, 1) {
		t.Fail()
	}

	a1 = m.GetAt(1, 1)
	if !dat.AtMatch(a1, 1, 1) {
		t.Fail()
	}

	a2 = m.GetAt(1, 2)
	if !dat.AtMatch(a2, 1, 2) {
		t.Fail()
	}
}

func TestMatrix_SetRow(t *testing.T) {

	dat := dataSetA

	m := NewMatrix(dat.Rows, dat.Cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m.SetRow(0, dat.RowData[0])
	r0 := m.GetRow(0)
	if !dat.RowMatch(r0, 0) {
		t.Fail()
	}

	m.SetRow(2, dat.RowData[2])
	r2 := m.GetRow(2)
	if !dat.RowMatch(r2, 2) {
		t.Fail()
	}

	m.Reset()

	dat = dataSetB

	m = NewMatrix(dat.Rows, dat.Cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m.SetRow(0, dat.RowData[0])
	r0 = m.GetRow(0)
	if !dat.RowMatch(r0, 0) {
		t.Fail()
	}

	m.SetRow(1, dat.RowData[1])
	r1 := m.GetRow(1)
	if !dat.RowMatch(r1, 1) {
		t.Fail()
	}
}

func TestMatrix_SetCol(t *testing.T) {

	dat := dataSetA

	m := NewMatrix(dat.Rows, dat.Cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m.SetCol(0, dat.ColData[0])
	c0 := m.GetCol(0)
	if !dat.ColMatch(c0, 0) {
		fmt.Println(c0)
		t.Fail()
	}

	m.SetCol(1, dat.ColData[1])
	c1 := m.GetCol(1)
	if !dat.ColMatch(c1, 1) {
		t.Fail()
	}

	m.Reset()

	dat = dataSetB

	m = NewMatrix(dat.Rows, dat.Cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m.SetCol(0, dat.ColData[0])
	c0 = m.GetCol(0)
	if !dat.ColMatch(c0, 0) {
		t.Fail()
	}

	m.SetCol(2, dat.ColData[2])
	c2 := m.GetCol(2)
	if !dat.ColMatch(c2, 2) {
		t.Fail()
	}
}

func TestMatrix_SetAt(t *testing.T) {

	dat := dataSetA

	m := NewMatrix(dat.Rows, dat.Cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m.SetAt(0, 1, dat.RowData[0][1])
	a0 := m.GetAt(0, 1)
	if !dat.AtMatch(a0, 0, 1) {
		t.Fail()
	}

	m.SetAt(1, 1, dat.RowData[1][1])
	a1 := m.GetAt(1, 1)
	if !dat.AtMatch(a1, 1, 1) {
		t.Fail()
	}

	m.SetAt(2, 1, dat.RowData[2][1])
	a2 := m.GetAt(2, 1)
	if !dat.AtMatch(a2, 2, 1) {
		t.Fail()
	}

	m.Reset()

	dat = dataSetB

	m = NewMatrix(dat.Rows, dat.Cols, nil)

	if m.IsEmpty() {
		t.Fail()
	}

	m.SetAt(0, 1, dat.RowData[0][1])
	a0 = m.GetAt(0, 1)
	if !dat.AtMatch(a0, 0, 1) {
		t.Fail()
	}

	m.SetAt(1, 1, dat.RowData[1][1])
	a1 = m.GetAt(1, 1)
	if !dat.AtMatch(a1, 1, 1) {
		t.Fail()
	}

	m.SetAt(1, 2, dat.RowData[1][2])
	a2 = m.GetAt(1, 2)
	if !dat.AtMatch(a2, 1, 2) {
		t.Fail()
	}

}

func TestMatrix_Apply(t *testing.T) {

	dat := dataSetA

	m := NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if m.IsEmpty() {
		t.Fail()
	}

	fmt.Println(m.Data)

	// if the cell we visit is a 2, make it an 8
	m.Apply(func(r, c int, cell DataType) DataType {
		if cell == 2 {
			return cell + 6
		}
		// otherwise, we just ignore it
		return cell
	})

	fmt.Println(m.Data)
}

func TestMatrix_Scale(t *testing.T) {

}

func TestMatrix_Scalar(t *testing.T) {

}

func TestMatrixCopyOf(t *testing.T) {

	dat := dataSetA

	mA := NewMatrix(dat.Rows, dat.Cols, dat.Data)

	if mA.IsEmpty() {
		t.Fail()
	}

	mB := CopyOf(mA)

	if mA.IsEmpty() {
		t.Fail()
	}

	if !(mA.Equal(mB) && mB.Equal(mA)) {
		t.Fail()
	}

	mA.SetAt(0, 1, 255.55)
	if mA.GetAt(0, 1) != 255.55 {
		t.Fail()
	}

	if mB.GetAt(0, 1) == 255.55 {
		t.Fail()
	}

}

func TestMatrixDotProduct(t *testing.T) {

	a := dataSetA

	mA := NewMatrix(a.Rows, a.Cols, a.Data)

	if mA.IsEmpty() {
		t.Fail()
	}

	b := dataSetB

	mB := NewMatrix(b.Rows, b.Cols, b.Data)

	if mB.IsEmpty() {
		t.Fail()
	}

	mC := Dot(mA, mB)

	mD := NewMatrix(mA.Rows, mB.Cols, dotProdSetASetBData)

	if !(mD.Equal(mC) && mC.Equal(mD)) {
		t.Fail()
	}
}

func TestMatrixAdd(t *testing.T) {

	mA := NewMatrix(3, 2, addSubSetA)

	if mA.IsEmpty() {
		t.Fail()
	}

	mB := NewMatrix(3, 2, addSubSetB)

	if mB.IsEmpty() {
		t.Fail()
	}

	mC := Add(mA, mB)

	mD := NewMatrix(3, 2, addSetASetBData)

	if !(mD.Equal(mC) && mC.Equal(mD)) {
		t.Fail()
	}

}

func TestMatrixSub(t *testing.T) {

	mA := NewMatrix(3, 2, addSubSetA)

	if mA.IsEmpty() {
		t.Fail()
	}

	mB := NewMatrix(3, 2, addSubSetB)

	if mB.IsEmpty() {
		t.Fail()
	}

	mC := Sub(mA, mB)

	mD := NewMatrix(3, 2, subSetASetBData)

	if !(mD.Equal(mC) && mC.Equal(mD)) {
		t.Fail()
	}

}

func TestMatrixMul(t *testing.T) {

}

func TestMatrixInv(t *testing.T) {

}
