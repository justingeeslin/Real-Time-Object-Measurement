import cv2
from ObjectMeasurer import ObjectMeasurer

# SHORT SIDE / X-AXIS FIRST
# A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in

# Open an image with an object you want to measure ontop of another, reference object of known dimensions
img = cv2.imread("../test-images/ucard-one-off-axis/ucard-one-off-axis.jpg")

# Construct the ObjectMeasurer with the size of the reference object
measurer = ObjectMeasurer(reference_size_mm=LETTER_MM)
# Get the measurements (in cm)
measurements = measurer.measure(img)

print(f"{[(m.width_cm, m.height_cm) for m in measurements]}")