package ocr

import (
	"log"
	"os"
	"time"

	"gocv.io/x/gocv"
)

type TextRecognizer struct {
	*PaddleModel
	batchNum int
	textLen  int
	shape    []int
	labels   []string

	mean    []float32
	scale   []float32
	isScale bool
}

func NewTextRecognizer(args *Config) *TextRecognizer {
	modelDir := args.RecModelDir
	labels := readLines2StringSlice(args.RecCharDictPath)
	if args.UseSpaceChar {
		labels = append(labels, " ")
	}
	rec := &TextRecognizer{
		PaddleModel: NewPaddleModel(args),
		batchNum:    args.RecBatchNum,
		textLen:     args.MaxTextLength,
		shape:       args.RecImageShape,
		labels:      labels,

		mean:    []float32{0.5, 0.5, 0.5},
		scale:   []float32{1 / 0.5, 1 / 0.5, 1 / 0.5},
		isScale: true,
	}
	if checkModelExists(modelDir) {
		home, _ := os.UserHomeDir()
		modelDir, _ = downloadModel(home+"/.paddleocr/rec/ch", modelDir)
	} else {
		log.Panicf("rec model path: %v not exist! Please check!", modelDir)
	}
	rec.LoadModel(modelDir)
	return rec
}

func (rec *TextRecognizer) Run(imgs []gocv.Mat, bboxes [][][]int) []OCRText {
	recResult := make([]OCRText, 0, len(imgs))
	batch := rec.batchNum
	var recTime int64
	c, imgH, imgW := rec.shape[0], rec.shape[1], rec.shape[2]
	for i := 0; i < len(imgs); i += batch {
		j := i + batch
		if len(imgs) < j {
			j = len(imgs)
		}

		maxWhRatio := float64(imgW) / float64(imgH)
		for k := i; k < j; k++ {
			h, w := imgs[k].Rows(), imgs[k].Cols()
			whRatio := float64(w) / float64(h)
			if whRatio > maxWhRatio {
				maxWhRatio = whRatio
			}
		}

		batchWidth := imgW
		normimgs := make([]float32, (j-i)*c*imgH*imgW)
		for k := i; k < j; k++ {
			srcimg := gocv.NewMat()
			imgs[k].CopyTo(&srcimg)
			resizeImg := crnnResize(srcimg, rec.shape, maxWhRatio)
			normalize(resizeImg, rec.mean, rec.scale, rec.isScale)

			data, _ := resizeImg.DataPtrFloat32()
			copy(normimgs[(k-i)*c*imgH*imgW:], data)
			if resizeImg.Cols() > batchWidth {
				batchWidth = resizeImg.Cols()
			}
		}

		st := time.Now()
		rec.input.Reshape([]int32{int32(j - i), int32(c), int32(imgH), int32(imgW)})
		rec.input.CopyFromCpu(normimgs)

		rec.predictor.Run()

		predictShape := rec.outputs[0].Shape()
		predictBatch := make([]float32, numElements(predictShape))
		rec.outputs[0].CopyToCpu(predictBatch)
		recTime += int64(time.Since(st).Milliseconds())

		for m := 0; m < int(predictShape[0]); m++ {
			var text string
			var count int
			var score float32
			var lastIndex int
			for n := 0; n < int(predictShape[1]); n++ {
				l := (m*int(predictShape[1]) + n) * int(predictShape[2])
				r := (m*int(predictShape[1]) + n + 1) * int(predictShape[2])
				// get argmax_idx & get score
				argmaxIdx, maxValue := argmax(predictBatch[l:r])
				if argmaxIdx > 0 && (!(n > 0 && argmaxIdx == lastIndex)) {
					score += maxValue
					count += 1
					text += rec.labels[argmaxIdx]
				}
				lastIndex = argmaxIdx
			}
			if score == 0.0 && count == 0 {
				continue
			}
			score /= float32(count)
			recResult = append(recResult, OCRText{
				BBox:  bboxes[i+m],
				Text:  text,
				Score: float64(score),
			})
		}
	}
	log.Println("rec num: ", len(recResult), ", rec time elapse: ", recTime, "ms")
	return recResult
}
