package ocr

import (
	"image"
	"image/color"
	"log"
	"math"
	"slices"
	"sort"

	"gocv.io/x/gocv"
)

// OCR is the OCR engine.
type OCR interface {
	Predict(img gocv.Mat) []Result
	ReadImage(name string) gocv.Mat
}

// Result is the OCR predict result.
// OCR result of a single image may contains multiple `Result`s,
type Result struct {
	Text      string    `json:"text"`      // Predicted text
	BBox      [][]int   `json:"bbox"`      // BBox box of the predicted text
	Score     float32   `json:"score"`     // Score of the predicted text
	Direction Direction `json:"direction"` // Direction of the predicted text (if classifier is enabled)
}

type impl struct {
	detector   *detector
	classifier *classifier
	recognizer *recognizer
}

// New creates a new OCR engine using the config file specified by `conf`.
func New(conf string) (OCR, error) {
	cfg, err := ReadConfig(conf)
	if err != nil {
		log.Panicf("Read config %v error: %v\n", conf, err)
	}

	detector, err := newDetector(cfg)
	if err != nil {
		return nil, err
	}
	recognizer, err := newRecognizer(cfg)
	if err != nil {
		return nil, err
	}
	classifier, err := newClassifier(cfg)
	if err != nil {
		return nil, err
	}

	return &impl{
		detector:   detector,
		recognizer: recognizer,
		classifier: classifier,
	}, nil
}

// Predict predicts the text in the image.
func (o *impl) Predict(img gocv.Mat) []Result {
	boxes := o.detector.Run(img)
	if len(boxes) == 0 {
		return nil
	}

	boxes = sortBoxes(boxes)
	dirs := make([]Direction, len(boxes))
	cropImgs := make([]gocv.Mat, len(boxes))
	for i, box := range boxes {
		cropImgs[i] = getRotateCropImage(img, box)
		defer cropImgs[i].Close()
	}
	if o.classifier != nil {
		cropImgs, dirs = o.classifier.run(cropImgs)
	}
	return o.recognizer.run(cropImgs, boxes, dirs)
}

// ReadImage reads the image into gocv.Mat from the file.
func (o *impl) ReadImage(name string) gocv.Mat {
	img := gocv.IMRead(name, gocv.IMReadColor)
	if img.Empty() {
		log.Panicf("Could not read image %s\n", name)
	}
	return img
}

func boxCompare(box1, box2 [][]int) bool {
	if box1[0][1] < box2[0][1] {
		return true
	}
	if box1[0][1] > box2[0][1] {
		return false
	}
	return box1[0][0] < box2[0][0]
}

func boxNeedSwap(box1, box2 [][]int) bool {
	return math.Abs(float64(box1[0][1]-box2[0][1])) < 10 && box1[0][0] < box2[0][0]
}

func sortBoxes(boxes [][][]int) [][][]int {
	sort.Slice(boxes, func(i, j int) bool {
		return boxCompare(boxes[i], boxes[j])
	})
	for i := 0; i < len(boxes)-1; i++ {
		if boxNeedSwap(boxes[i+1], boxes[i]) {
			boxes[i], boxes[i+1] = boxes[i+1], boxes[i]
		}
	}
	return boxes
}

// https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/cpp_infer/src/utility.cpp#L133
func getRotateCropImage(srcImg gocv.Mat, box [][]int) gocv.Mat {
	xCollect := []int{box[0][0], box[1][0], box[2][0], box[3][0]}
	yCollect := []int{box[0][1], box[1][1], box[2][1], box[3][1]}

	left, right := slices.Min(xCollect), slices.Max(xCollect)
	top, bottom := slices.Min(yCollect), slices.Max(yCollect)
	cropImg := srcImg.Region(image.Rect(left, top, right, bottom))
	defer cropImg.Close()

	pts := make([][]int, len(box))
	for i, pt := range box {
		pts[i] = []int{pt[0] - left, pt[1] - top}
	}

	cropW := int(math.Sqrt(math.Pow(float64(box[0][0]-box[1][0]), 2) + math.Pow(float64(box[0][1]-box[1][1]), 2)))
	cropH := int(math.Sqrt(math.Pow(float64(box[0][0]-box[3][0]), 2) + math.Pow(float64(box[0][1]-box[3][1]), 2)))

	m := gocv.GetPerspectiveTransform(
		gocv.NewPointVectorFromPoints([]image.Point{
			image.Pt(pts[0][0], pts[0][1]),
			image.Pt(pts[1][0], pts[1][1]),
			image.Pt(pts[2][0], pts[2][1]),
			image.Pt(pts[3][0], pts[3][1]),
		}),
		gocv.NewPointVectorFromPoints([]image.Point{
			image.Pt(0, 0),
			image.Pt(cropW, 0),
			image.Pt(cropW, cropH),
			image.Pt(0, cropH),
		}),
	)
	defer m.Close()

	dstImg := gocv.NewMat()
	gocv.WarpPerspectiveWithParams(cropImg, &dstImg, m, image.Pt(cropW, cropH), gocv.InterpolationLinear, gocv.BorderReplicate, color.RGBA{0, 0, 0, 0})
	if float64(dstImg.Rows()) >= float64(dstImg.Cols())*1.5 {
		gocv.Transpose(dstImg, &dstImg)
		gocv.Flip(dstImg, &dstImg, 0)
	}
	return dstImg
}
