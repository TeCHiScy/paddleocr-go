package ocr

import (
	"cmp"
	"os"
	"sort"

	"gocv.io/x/gocv"
)

func clamp[T cmp.Ordered](x, min, max T) T {
	if x > max {
		return max
	}
	if x < min {
		return min
	}
	return x
}

func argmax[T cmp.Ordered](s []T) (int, T) {
	max, idx := s[0], 0
	for i, v := range s {
		if v > max {
			idx, max = i, v
		}
	}
	return idx, max
}

func isPathExist(path string) bool {
	if _, err := os.Stat(path); err != nil {
		return false
	}
	return true
}

func accumulate(vals []int32) int32 {
	n := int32(1)
	for _, v := range vals {
		n *= v
	}
	return n
}

// https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/cpp_infer/src/preprocess_op.cpp#L40
func normalize(im gocv.Mat, mean []float32, scale []float32, isScale bool) {
	e := float32(1.0)
	if isScale {
		e /= 255.0
	}
	im.ConvertToWithParams(&im, gocv.MatTypeCV32FC3, e, 0)
	bgrChannels := gocv.Split(im)
	for i := range bgrChannels {
		defer bgrChannels[i].Close()
		bgrChannels[i].ConvertToWithParams(&bgrChannels[i], gocv.MatTypeCV32FC1, 1.0*scale[i], (0.0-mean[i])*scale[i])
	}
	gocv.Merge(bgrChannels, &im)
}

// https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/cpp_infer/src/preprocess_op.cpp#L19
func permute(im gocv.Mat) []float32 {
	rh := im.Rows()
	rw := im.Cols()
	rc := im.Channels()
	data := make([]float32, rh*rw*rc)
	for i := 0; i < rc; i++ {
		t := gocv.NewMatWithSize(rh, rw, gocv.MatTypeCV32FC1)
		defer t.Close()
		gocv.ExtractChannel(im, &t, i)
		x, _ := t.DataPtrFloat32()
		copy(data[i*rh*rw:], x)
	}
	return data
}

// https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/cpp_infer/src/preprocess_op.cpp#L28
func permuteBatch(imgs []gocv.Mat) []float32 {
	var data []float32
	for _, img := range imgs {
		data = append(data, permute(img)...)
	}
	return data
}

type Slice struct {
	sort.Interface
	idx []int
}

func (s Slice) Swap(i, j int) {
	s.Interface.Swap(i, j)
	s.idx[i], s.idx[j] = s.idx[j], s.idx[i]
}

func NewSlice(n sort.Interface) *Slice {
	s := &Slice{Interface: n, idx: make([]int, n.Len())}
	for i := range s.idx {
		s.idx[i] = i
	}
	return s
}

func NewIntSlice(n ...int) *Slice         { return NewSlice(sort.IntSlice(n)) }
func NewFloat64Slice(n ...float64) *Slice { return NewSlice(sort.Float64Slice(n)) }
func NewStringSlice(n ...string) *Slice   { return NewSlice(sort.StringSlice(n)) }

type xIntSortBy [][]int

func (a xIntSortBy) Len() int           { return len(a) }
func (a xIntSortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a xIntSortBy) Less(i, j int) bool { return a[i][0] < a[j][0] }

type xFloatSortBy [][]float32

func (a xFloatSortBy) Len() int           { return len(a) }
func (a xFloatSortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a xFloatSortBy) Less(i, j int) bool { return a[i][0] < a[j][0] }
