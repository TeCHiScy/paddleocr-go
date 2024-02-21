package ocr

import (
	"image"
	"image/color"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"gocv.io/x/gocv"
)

type recognizer struct {
	*Predictor
	batchNum int
	textLen  int
	shape    []int
	labels   []string

	mean    []float32
	scale   []float32
	isScale bool
}

// newRecognizer creates a new text recognizer.
func newRecognizer(cfg *Config) (*recognizer, error) {
	rcfg := cfg.Recognizer
	model, err := NewPredictor(&cfg.Predictor, rcfg.ModelDir)
	if err != nil {
		return nil, err
	}
	return &recognizer{
		Predictor: model,
		batchNum:  rcfg.BatchNum,
		textLen:   rcfg.MaxTextLength,
		shape:     rcfg.ImageShape,
		labels:    readDict(rcfg.CharDictPath),

		mean:    []float32{0.5, 0.5, 0.5},
		scale:   []float32{1 / 0.5, 1 / 0.5, 1 / 0.5},
		isScale: true,
	}, nil
}

func readDict(filepath string) []string {
	data, err := os.ReadFile(filepath)
	if err != nil {
		log.Println("read recognizer key file error!")
		return nil
	}
	labels := strings.Split(string(data), "\n")
	labels = append([]string{"#"}, labels...) // blank char for ctc
	labels = append(labels, " ")
	return labels
}

func (p *recognizer) run(imgs []gocv.Mat, bboxes [][][]int, dirs []Direction) []Result {
	t := time.Now()
	h, w := p.shape[1], p.shape[2]

	widths := make([]float64, 0, len(imgs))
	for _, img := range imgs {
		widths = append(widths, float64(img.Cols())/float64(img.Rows()))
	}

	s := NewFloat64Slice(widths...)
	sort.Sort(s)

	results := make([]Result, len(imgs))
	for i := 0; i < len(imgs); i += p.batchNum {
		j := min(i+p.batchNum, len(imgs))
		batchNum := j - i

		maxWhRatio := float64(w) / float64(h)
		for k := i; k < j; k++ {
			maxWhRatio = max(maxWhRatio, float64(imgs[s.idx[k]].Cols())/float64(imgs[s.idx[k]].Rows()))
		}

		batchWidth := w
		normImgs := make([]gocv.Mat, 0, batchNum)
		for k := i; k < j; k++ {
			resizeImg := p.resize(imgs[s.idx[k]], p.shape, maxWhRatio)
			defer resizeImg.Close()

			normalize(resizeImg, p.mean, p.scale, p.isScale)
			normImgs = append(normImgs, resizeImg)
			batchWidth = max(batchWidth, resizeImg.Cols())
		}

		p.input.Reshape([]int32{int32(batchNum), 3, int32(h), int32(batchWidth)})
		p.input.CopyFromCpu(permuteBatch(normImgs))
		p.predictor.Run()

		shape := p.output.Shape()
		predicts := make([]float32, accumulate(shape))
		p.output.CopyToCpu(predicts)

		for m := 0; m < int(shape[0]); m++ {
			var (
				text      string
				count     int
				sumScore  float32
				lastIndex int
			)
			for n := 0; n < int(shape[1]); n++ {
				l := (m*int(shape[1]) + n) * int(shape[2])
				r := (m*int(shape[1]) + n + 1) * int(shape[2])
				argmaxIdx, maxValue := argmax(predicts[l:r])
				if argmaxIdx > 0 && (!(n > 0 && argmaxIdx == lastIndex)) {
					count += 1
					sumScore += maxValue
					text += p.labels[argmaxIdx]
				}
				lastIndex = argmaxIdx
			}
			var score float32
			if count > 0 {
				score = sumScore / float32(count)
			}
			results[s.idx[i+m]] = Result{
				Text:      text,
				Direction: dirs[s.idx[i+m]],
				BBox:      bboxes[s.idx[i+m]],
				Score:     score,
			}
		}
	}
	log.Printf("recognizer: box num: %d, elapsed: %dms\n", len(results), time.Since(t).Milliseconds())
	return results
}

func (p *recognizer) resize(img gocv.Mat, shape []int, whRatio float64) gocv.Mat {
	imgH := shape[1]
	imgW := int(float64(imgH) * whRatio)
	ratio := float64(img.Cols()) / float64(img.Rows())

	resizeW := int(math.Ceil(float64(imgH) * ratio))
	if resizeW > imgW {
		resizeW = imgW
	}

	resizeImg := gocv.NewMat()
	gocv.Resize(img, &resizeImg, image.Pt(resizeW, imgH), 0, 0, gocv.InterpolationLinear)
	gocv.CopyMakeBorder(resizeImg, &resizeImg, 0, 0, 0, imgW-resizeImg.Cols(), gocv.BorderConstant, color.RGBA{0, 0, 0, 0})
	return resizeImg
}
