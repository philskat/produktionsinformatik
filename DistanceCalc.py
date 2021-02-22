
import numpy as np
import cv2 as cv

def show_wait_destroy(winname, img):
    cv.imshow(winname, cv.resize(img, None, fx=0.25, fy=0.25))
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

#Calculates mean values for numbers that are close to each other
def reduceneighbours(array):
    output = []
    group = []
    for number in array:
        if len(group) == 0:
            group.append(number)
        if number - group[-1] > 5:
            output.append(round(np.sum(group)/len(group)))
            group.clear()
        group.append(number)
        if number == array[-1]:
            output.append(round(np.sum(group)/len(group)))
            group.clear()
    return output

def main():
    
    # Load the image
    src = cv.imread("assets/Img18.jpg", cv.IMREAD_COLOR)
    cv.imshow("src", cv.resize(src, None, fx=0.25, fy=0.25))

    #Transform image to gray
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #show_wait_destroy("gray", gray)

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    #show_wait_destroy("binary", bw)

    # Create the images that will use to extract the vertical lines
    vertical = np.copy(bw)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    # Show extracted vertical lines
    #show_wait_destroy("vertical", vertical)

    # Inverse vertical image
    vertical = cv.bitwise_not(vertical)
    #show_wait_destroy("vertical_bit", vertical)
    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''
    # Step 1
    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 3, -2)
    #show_wait_destroy("edges", edges)
    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    #show_wait_destroy("dilate", edges)
    # Step 3
    smooth = np.copy(vertical)
    # Step 4
    smooth = cv.blur(smooth, (2, 2))
    # Step 5
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]

    # Get colums with lines
    columns = np.where(vertical == 0)[1]
    columns = reduceneighbours(np.unique(columns))

    #incorrect Detection
    if len(columns) != 4:
        print("Object could not be detected")
        return 0
    
    #Print Lines
    imageheigt = src.shape[0]
    for column in columns:
        cv.line(src, (column,0), (column, src.shape[0]), (0,255,0), 3)
    
    cv.line(src, (columns[0], round(imageheigt/2+400)), (columns[3], round(imageheigt/2+400)), (255,0,0), 3)
    cv.line(src, (columns[1], round(imageheigt/2)-400), (columns[2], round(imageheigt/2)-400), (255,0,0), 3)

    d1 = columns[3]-columns[0]
    d2 = columns[2]-columns[1]

    sizeOnSensord1 = (4.23 * d1) / 3000
    sizeOfObjectd1 = (0.3 * sizeOnSensord1) / 4.36
    sizeOnSensord2 = (4.23 * d2) / 3000
    sizeOfObjectd2 = (0.3 * sizeOnSensord2) / 4.36
    cv.putText((src), str(round(sizeOfObjectd1*100,4)), (round(columns[0]+d1/2),round(imageheigt/2+350)), cv.FONT_ITALIC, (5), (255,0,0), 5, cv.LINE_AA)
    cv.putText((src), str(round(sizeOfObjectd2*100,4)), (round(columns[1]+d2/2),round(imageheigt/2-450)), cv.FONT_ITALIC, (5), (255,0,0), 5, cv.LINE_AA)
    show_wait_destroy("Line", src)

    return 0
if __name__ == "__main__":
    main()