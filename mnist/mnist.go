package mnist

import (
	"fmt"
	"os"
	"path/filepath"
)

const (
	TrainImagesFile = "train-images-idx3-ubyte"
	TrainLabelsFile = "train-labels-idx1-ubyte"
	TestImagesFile  = "t10k-images-idx3-ubyte"
	TestLabelsFile  = "t10k-labels-idx1-ubyte"

	labelsFileMagic = 0x00000801
	imagesFileMagic = 0x00000803

	msgInvalidFormat = "Invalid format: %s"
	msgSizeUnmatch   = "Size unmatch"
)

func fileError(f *os.File) error {
	return fmt.Errorf(msgInvalidFormat, f.Name())
}

//readInt32 takes a file as an input and reads the bytes,
//converting 4 bytes at a time to a big endian integer
func readInt32(f *os.File) (int, error) {
	buf := make([]byte, 4)
	n, err := f.Read(buf)
	if err != nil {
		return 0, err
	} else if n != 4 {
		return 0, fileError(f)
	}
	v := 0
	for _, x := range buf {
		v = v*256 + int(x)
	}
	return v, nil
}

//imageData holds the information about an image
type imageData struct {
	N      int //I don't have a clue what this number represents
	Width  int
	Height int
	Data   []uint8
}

func readImagesFile(path string) (*imageData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	magic, err := readInt32(f)
	if err != nil || magic != imagesFileMagic {
		return nil, fileError(f)
	}
	n, err := readInt32(f)
	if err != nil {
		return nil, fileError(f)
	}
	width, err := readInt32(f)
	if err != nil {
		return nil, err
	}
	height, err := readInt32(f)
	if err != nil {
		return nil, fileError(f)
	}
	size := n * width * height
	data := &imageData{
		N:      n,
		Width:  width,
		Height: height,
		Data:   make([]uint8, size),
	}
	length, err := f.Read(data.Data) //I don't know what this does
	if err != nil || length != size {
		return nil, fileError(f)
	}
	return data, nil
}

type labelData struct {
	N    int
	Data []uint8
}

func readLabelsFile(path string) (*labelData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	magic, err := readInt32(f)
	if err != nil || magic != labelsFileMagic {
		return nil, fileError(f)
	}
	n, err := readInt32(f)
	if err != nil {
		return nil, fileError(f)
	}
	data := &labelData{
		N:    n,
		Data: make([]uint8, n),
	}
	len, err := f.Read(data.Data)
	if err != nil || len != n {
		return nil, fileError(f)
	}
	return data, nil
}

//What? I don't know
type DigitImage struct {
	Digit int
	Image [][]uint8
}

//DataSet does what?
type DataSet struct {
	N      int
	Width  int
	Height int
	Data   []DigitImage
}

func ReadDataSet(imagesPath, labelsPath string) (*DataSet, error) {
	images, err := readImagesFile(imagesPath)
	if err != nil {
		return nil, err
	}
	labels, err := readLabelsFile(labelsPath)
	if err != nil {
		return nil, err
	}
	if images.N != labels.N {
		return nil, fmt.Errorf("%v, %v, %v, ", msgSizeUnmatch, labelsPath, imagesPath)
	}
	dataSet := &DataSet{
		N:      images.N,
		Width:  images.Width,
		Height: images.Height,
		Data:   make([]DigitImage, images.N),
	}
	rows := splitToRows(images.Data, images.N, images.Height)
	for i := 0; i < dataSet.N; i++ {
		data := &dataSet.Data[i]
		data.Digit = int(labels.Data[i])
		data.Image = rows[0:dataSet.Height]
		rows = rows[dataSet.Height:]
	}
	return dataSet, nil
}

func ReadTrainSet(dir string) (*DataSet, error) {
	imagesPath := filepath.Join(dir, TrainImagesFile)
	labelsPath := filepath.Join(dir, TrainLabelsFile)
	return ReadDataSet(imagesPath, labelsPath)
}

func ReadTestSet(dir string) (*DataSet, error) {
	imagesPath := filepath.Join(dir, TestImagesFile)
	labelsPath := filepath.Join(dir, TestLabelsFile)
	return ReadDataSet(imagesPath, labelsPath)
}

func splitToRows(data []uint8, N, H int) [][]uint8 {
	numRows := N * H
	rows := make([][]uint8, numRows)
	for i := 0; i < numRows; i++ {
		rows[i] = data[0:H]
		data = data[H:]
	}
	return rows
}

func PrintImage(image [][]uint8) {
	for _, row := range image {
		for _, pix := range row {
			if pix == 0 {
				fmt.Print(" ")
			} else {
				fmt.Printf("%X", pix/16)
			}
		}
		fmt.Println()
	}
}
