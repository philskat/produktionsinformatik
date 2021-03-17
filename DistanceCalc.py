from tkinter import *
from configparser import ConfigParser
import numpy as np
import cv2 as cv

image_path = "assets/Img15.jpg"
#outer_diameter = 0.075
outer_diameter = 0
inner_diameter = 0

# Load config from config.ini
def loadConfig():
    global sensor_width
    global focal_length
    global distance_object

    config = ConfigParser()
    config.read('config.ini')

    sensor_width = config["Default"].getfloat("sensor_width")
    focal_length = config["Default"].getfloat("focal_length")
    distance_object = config["Default"].getfloat("distance_object")

# Handle the click of submit on input window
def handleClick(outerInput, innerInput, master):
    global outer_diameter
    global inner_diameter
    try:
        outer_diameter = float(outerInput.get()) / 10000
        inner_diameter = float(innerInput.get()) / 10000
        master.destroy()

    except ValueError:
        print("Wrong Values")

# Shows given image in a window and wait for key press
def show_wait_destroy(winname, img):
    cv.imshow(winname, cv.resize(img, None, fx=0.25, fy=0.25))
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

# Calculates mean values for numbers that are close to each other
def reduceNeighbours(array):
    output = []
    group = [ array[0] ]

    for number in array[1:]:
        if number - group[-1] > 5:
            output.append(round(np.sum(group)/len(group)))
            group.clear()

        group.append(number)

        if number == array[-1]:
            output.append(round(np.sum(group)/len(group)))
            group.clear()
    return output

# Calculates the real word size from the amout of pixels
def pixelSizeToRealWorld(distanceInPixels, imageWidth):
    sizeOnSensor = (sensor_width * distanceInPixels) / imageWidth
    return ((distance_object + outer_diameter / 2 ) * sizeOnSensor) / focal_length


def main():
    # Load config for measuring values
    loadConfig()
    
    # Load and show the image
    src = cv.imread(image_path)
    #show_wait_destroy("Src", src)

    cv.imshow("Original", cv.resize(src, None, fx=0.25, fy=0.25))

    # Setup input window
    master = Tk()
    master.title("Diameters input")
    Label(master, text="Outer diameter (mm)").grid(row=0)
    Label(master, text="Inner diameter (mm)").grid(row=1)

    outerInput = Entry(master)
    innerInput = Entry(master)

    outerInput.grid(row=0, column=1)
    innerInput.grid(row=1, column=1)

    Button(master, 
           text="Submit", 
           command=lambda: handleClick(outerInput, innerInput, master)
    ).grid(row=3)

    # Run input window
    mainloop()

    cv.destroyAllWindows()

    # Transform image to gray
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #show_wait_destroy("gray", gray)
    height, width = gray.shape

    # Create binary image with adaptive Threshold
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    #show_wait_destroy("binary", bw)

    # Create the images that will use to extract the vertical lines
    vertical = bw

    # Specify size on vertical axis
    verticalsize = height // 30

    # Create structure element for vertical line
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))

    # Apply structure on image
    vertical = cv.erode(vertical, verticalStructure)

    # Show extracted vertical lines
    #show_wait_destroy("vertical", vertical)

    # Get colums with vertical structure
    columns = np.where(vertical == 255)[1]
    columns = reduceNeighbours(np.unique(columns))

    # Object must have 4 lines
    if len(columns) != 4:
        print("Object could not be detected")
        return 0
    
    # Print Lines to the image
    for column in columns:
        cv.line(src, (column,0), (column, height), (0,255,0), 3)
    
    # Draw diameter
    cv.line(src, (columns[1], height // 3), (columns[2], height // 3), (255,0,0), 3)
    cv.line(src, (columns[0], 2 * height // 3), (columns[3], 2 * height // 3), (255,0,0), 3)

    # Calculate inner and outer Diameter in pixels
    outerDiameter = columns[3]-columns[0]
    innerDiameter = columns[2]-columns[1]

    # Convert pixel size to real world size
    realOuterDiameter = pixelSizeToRealWorld(outerDiameter, width)
    realInnerDiameter = pixelSizeToRealWorld(innerDiameter, width)

    # Write value of real diameter on the image
    cv.putText((src), str(round(realInnerDiameter*100,2)) + "cm", (round(columns[1]+innerDiameter/2), height // 3 - 25), cv.FONT_ITALIC, 4, (255,0,0), 5, cv.LINE_AA)
    cv.putText((src), str(round(realOuterDiameter*100,2)) + "cm", (round(columns[0]+outerDiameter/2), 2 * height // 3 - 25), cv.FONT_ITALIC, 4, (255,0,0), 5, cv.LINE_AA)
    show_wait_destroy("Line", src)

    return 0

if __name__ == "__main__":
    main()