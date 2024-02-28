package main

import (
	"flag"
	"log"
	"path/filepath"

	"github.com/TeCHiScy/paddleocr-go/ocr"
)

func main() {
	var conf, image, imageDir string
	flag.StringVar(&conf, "config", "config/conf.yaml", "config for ocr engine.")
	flag.StringVar(&image, "image", "", "image to predict. if not given, will use image_dir.")
	flag.StringVar(&imageDir, "image_dir", "", "imgs in dir to be predicted.")
	flag.Parse()

	o, err := ocr.New(conf)
	if err != nil {
		log.Panicf("create ocr error: %+v", err)
	}

	if image != "" {
		results := o.Predict(o.ReadImage(image))
		for _, res := range results {
			log.Println(res)
		}
		return
	}

	if imageDir != "" {
		names := []string{}
		jpgs, _ := filepath.Glob(imageDir + "/*.jpg")
		names = append(names, jpgs...)
		pngs, _ := filepath.Glob(imageDir + "/*.png")
		names = append(names, pngs...)

		for _, name := range names {
			results := o.Predict(o.ReadImage(name))
			log.Printf("======== image: %v =======\n", name)
			for _, res := range results {
				log.Println(res)
			}
		}
	}
}
