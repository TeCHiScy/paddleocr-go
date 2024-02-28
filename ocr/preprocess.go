package ocr

import (
	"image"
	"image/color"
	"math"

	"gocv.io/x/gocv"
)

func resizeByShape(img gocv.Mat, resizeShape []int) (gocv.Mat, int, int) {
	resizeH := resizeShape[0]
	resizeW := resizeShape[1]
	gocv.Resize(img, &img, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)
	return img, resizeH, resizeW
}

func resizeByMaxLen(img gocv.Mat, maxLen int) (gocv.Mat, int, int) {
	oriH := img.Rows()
	oriW := img.Cols()
	var resizeH, resizeW int = oriH, oriW

	var ratio float64 = 1.0
	if resizeH > maxLen || resizeW > maxLen {
		if resizeH > resizeW {
			ratio = float64(maxLen) / float64(resizeH)
		} else {
			ratio = float64(maxLen) / float64(resizeW)
		}
	}

	resizeH = int(float64(resizeH) * ratio)
	resizeW = int(float64(resizeW) * ratio)

	if resizeH%32 == 0 {
		resizeH = resizeH
	} else if resizeH/32 <= 1 {
		resizeH = 32
	} else {
		resizeH = (resizeH/32 - 1) * 32
	}

	if resizeW%32 == 0 {
		resizeW = resizeW
	} else if resizeW/32 <= 1 {
		resizeW = 32
	} else {
		resizeW = (resizeW/32 - 1) * 32
	}

	if resizeW <= 0 || resizeH <= 0 {
		return gocv.NewMat(), 0, 0
	}

	gocv.Resize(img, &img, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)
	return img, resizeH, resizeW
}

func normPermute(img gocv.Mat, mean []float32, std []float32, scaleFactor float32) []float32 {
	img.ConvertTo(&img, gocv.MatTypeCV32F)
	img.DivideFloat(scaleFactor)
	defer img.Close()

	c := gocv.Split(img)
	data := make([]float32, img.Rows()*img.Cols()*img.Channels())
	for i := 0; i < 3; i++ {
		c[i].SubtractFloat(mean[i])
		c[i].DivideFloat(std[i])
		defer c[i].Close()
		x, _ := c[i].DataPtrFloat32()
		copy(data[i*img.Rows()*img.Cols():], x)
	}
	return data
}

type DetPreProcess interface {
	Run(gocv.Mat) ([]float32, int, int)
}

type DBPreProcess struct {
	resizeType  int
	imageShape  []int
	maxSideLen  int
	mean        []float32
	std         []float32
	scaleFactor float32
}

func NewDBProcess(shape []int, sideLen int) *DBPreProcess {
	db := &DBPreProcess{
		resizeType:  0,
		imageShape:  shape,
		maxSideLen:  sideLen,
		mean:        []float32{0.485, 0.456, 0.406},
		std:         []float32{0.229, 0.224, 0.225},
		scaleFactor: 255.0,
	}
	if len(shape) > 0 {
		db.resizeType = 1
	}
	if sideLen == 0 {
		db.maxSideLen = 2400
	}
	return db
}

func (d *DBPreProcess) Run(img gocv.Mat) ([]float32, int, int) {
	var resizeH, resizeW int
	if d.resizeType == 0 {
		img, resizeH, resizeW = resizeByMaxLen(img, d.maxSideLen)
	} else {
		img, resizeH, resizeW = resizeByShape(img, d.imageShape)
	}

	im := normPermute(img, d.mean, d.std, d.scaleFactor)
	return im, resizeH, resizeW
}

func clsResize(img gocv.Mat, resizeShape []int) gocv.Mat {
	imgH, imgW := resizeShape[1], resizeShape[2]
	h, w := img.Rows(), img.Cols()
	ratio := float64(w) / float64(h)
	var resizeW int
	if math.Ceil(float64(imgH)*ratio) > float64(imgW) {
		resizeW = imgW
	} else {
		resizeW = int(math.Ceil(float64(imgH) * ratio))
	}
	gocv.Resize(img, &img, image.Pt(resizeW, imgH), 0, 0, gocv.InterpolationLinear)
	if resizeW < imgW {
		gocv.CopyMakeBorder(img, &img, 0, 0, 0, imgW-resizeW, gocv.BorderConstant, color.RGBA{0, 0, 0, 0})
	}
	return img
}

func crnnResize(img gocv.Mat, recImageShape []int, whRatio float64) gocv.Mat {
	imgH, imgW := recImageShape[1], recImageShape[2]
	imgW = int(float64(imgH) * whRatio)
	ratio := float64(img.Cols()) / float64(img.Rows())

	var resizeW int
	if math.Ceil(float64(imgH)*ratio) > float64(imgW) {
		resizeW = imgW
	} else {
		resizeW = int(math.Ceil(float64(imgH) * ratio))
	}

	resizeImg := gocv.NewMat()
	gocv.Resize(img, &resizeImg, image.Pt(resizeW, imgH), 0, 0, gocv.InterpolationLinear)
	gocv.CopyMakeBorder(resizeImg, &resizeImg, 0, 0, 0, imgW-resizeImg.Cols(), gocv.BorderConstant, color.RGBA{0, 0, 0, 0})
	return resizeImg
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
		gocv.ExtractChannel(im, &t, i)
		x, _ := t.DataPtrFloat32()
		copy(data[i*rh*rw:], x)
	}
	return data
}
