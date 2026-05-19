from utils import Utils,imagePlot,WordCorpus
from ultralytics import YOLO
def main():
    img = Utils.read_image('images/testml4.jpg')
    Words = WordCorpus()
    hist = Words.histogram(Utils.gray_scale(img))
    cropped = Words.crop_text(Utils.gray_scale(img), hist)
    print(f"Number of lines detected: {len(cropped)}")
    for i in cropped:
        deionised = Utils.bilateral_filter(i, d=9, sigma_color=75, sigma_space=75)
        clahe = Utils.clahe(deionised, clip_limit=2.0, tile_grid_size=(9,9))
        # threshold = Utils.threshold(clahe, max_value=255, block_size=21, C=7)
        # morphological = Utils.morphological_operations(clahe, kernel_size=(3, 3), iterations=1)
        
        #imagePlot.show_image(clahe, title="Processed Image")
        imagePlot.plot_images([i, deionised, clahe], titles=["Original", "Deionised", "CLAHE"]) 
        text = Utils.ocr(clahe)
        print(text)
if __name__ == "__main__":
    main()

    