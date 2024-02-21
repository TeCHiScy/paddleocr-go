package ocr

import (
	"image"
	"image/color"
	"log"
	"math"
	"time"

	"gocv.io/x/gocv"
)

// classifier is the text classifier.
type classifier struct {
	*Predictor
	batchNum int
	thresh   float32
	shape    []int

	mean    []float32
	scale   []float32
	isScale bool
}

// Direction is the classifier result.
type Direction struct {
	Label int     // Predicted label
	Score float32 // Score of the predicted label
}

// newClassifier creates a new text classifier.
func newClassifier(cfg *Config) (*classifier, error) {
	ccfg := cfg.Classifier
	if !ccfg.Enabled {
		return nil, nil
	}
	model, err := NewPredictor(&cfg.Predictor, ccfg.ModelDir)
	if err != nil {
		return nil, err
	}
	return &classifier{
		Predictor: model,
		thresh:    ccfg.Thresh,
		batchNum:  ccfg.BatchNum,
		shape:     ccfg.ImageShape,

		mean:    []float32{0.5, 0.5, 0.5},
		scale:   []float32{1 / 0.5, 1 / 0.5, 1 / 0.5},
		isScale: true,
	}, nil
}

func (p *classifier) run(imgs []gocv.Mat) ([]gocv.Mat, []Direction) {
	t := time.Now()
	directions := make([]Direction, len(imgs))
	c, h, w := p.shape[0], p.shape[1], p.shape[2]
	for i := 0; i < len(imgs); i += p.batchNum {
		j := min(i+p.batchNum, len(imgs))

		normImgs := []gocv.Mat{}
		for k := i; k < j; k++ {
			resizeImg := p.resize(imgs[k], p.shape)
			defer resizeImg.Close()

			normalize(resizeImg, p.mean, p.scale, p.isScale)
			if resizeImg.Cols() < p.shape[2] {
				gocv.CopyMakeBorder(resizeImg, &resizeImg, 0, 0, 0, p.shape[2]-resizeImg.Cols(), gocv.BorderConstant, color.RGBA{0, 0, 0, 0})
			}
			normImgs = append(normImgs, resizeImg)
		}

		p.input.Reshape([]int32{int32(j - i), int32(c), int32(h), int32(w)})
		p.input.CopyFromCpu(permuteBatch(normImgs))
		p.predictor.Run()

		shape := p.output.Shape()
		predicts := make([]float32, accumulate(shape))
		p.output.CopyToCpu(predicts)

		for m := 0; m < int(shape[0]); m++ {
			l := m * int(shape[1])
			r := (m + 1) * int(shape[1])
			label, score := argmax(predicts[l:r])
			directions[i+m] = Direction{Score: score, Label: label}
		}
	}

	for i, dir := range directions {
		if dir.Label%2 == 1 && dir.Score > p.thresh {
			gocv.Rotate(imgs[i], &imgs[i], gocv.Rotate180Clockwise)
		}
	}

	log.Printf("classifier: box num: %d, elapsed: %dms\n", len(directions), time.Since(t).Milliseconds())
	return imgs, directions
}

func (p *classifier) resize(img gocv.Mat, resizeShape []int) gocv.Mat {
	imgH, imgW := resizeShape[1], resizeShape[2]
	h, w := img.Rows(), img.Cols()
	ratio := float64(w) / float64(h)
	var resizeW int
	if math.Ceil(float64(imgH)*ratio) > float64(imgW) {
		resizeW = imgW
	} else {
		resizeW = int(math.Ceil(float64(imgH) * ratio))
	}
	resizeImg := gocv.NewMat()
	gocv.Resize(img, &resizeImg, image.Pt(resizeW, imgH), 0, 0, gocv.InterpolationLinear)
	return resizeImg
}
