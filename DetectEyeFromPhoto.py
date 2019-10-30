import numpy as np
import matplotlib.pyplot as plt
import cv2, os, math

def FrameCapture(path):
    # Path to video file
    cap = cv2.VideoCapture(path)
    #print(cap.isOpened())
    frames = []
    count = 0
    while(cap.isOpened() and count <=50):
        ret, frame = cap.read()
        frames.append(frame)
        count += 1
        #print(count)#print(type(frame))
    cap.release()
    print('done reading in video')
    return(frames)
#TODO: finish cropping return statement
def preprocess(imgs):
    """
    preprocesses the images from cv2 numpy arrays to send to training network
    """
    newarr = []
    for image in imgs:
        if image is None:
            print('no image found')
        else:
            ret,frame=cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), (5,5),0),25,255,cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/64, param1=200, param2=10, minRadius=20, maxRadius=50)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                bestc = findcentercircle(image,circles)
                # Makes the entire image black
                _, black_frame = cv2.threshold(frame, 255, 255, cv2.THRESH_BINARY)
                # Puts a white circle the size of the interference pattern
                cv2.circle(black_frame, (bestc[0], bestc[1]), bestc[2]*6, (255, 255, 255), -1)
                # Make everything black, return the white portion of the image back to its normal state
                res = cv2.bitwise_and(image,image, mask= black_frame)

                # Generate a rectangle encompassing the intererence pattern
                # Cropping begins from the upper left corner of the rectangle; hence the subtracting of radiuses to make the
                # Starting point of the cropping the the upper left corner; then it adds the diameter of the interference pattern
                x_center = bestc[0]
                y_center = bestc[1]
                radius = bestc[2]
                # Crop the image
                crop_res = res[y_center - radius*6 : y_center + radius*6, x_center - radius*6: x_center + radius*6]
                crop_res  = cv2.resize(crop_res, (500, 500))
                # Saturate the image
                hsv = cv2.cvtColor(crop_res, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                s = s*2
                final_hsv = cv2.merge((h, s, v))
                saturated_res = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            newarr.append(saturated_res)
    return(newarr)

def findcentercircle(img,circles):
    min_x = 0
    min_y = 0

    for c in circles[0,:]:
        #print(c[0])

        if (c[0] >= min_x and c[1] >= min_y):
            min_x = c[0]
            min_y = c[1]
            #print("yes")
            best_circle = c

    return(best_circle)

def main():
    path = os.path.abspath('preprocess')
    files = []

    for r, d, f in os.walk(path):
        for file in f:
            if '.avi' in file:
                files.append(os.path.join(r, file))
    for f in files:
        print(f)

    frames = FrameCapture(files[0])
    rames = frames[:11]
    processed = preprocess(frames)
    for im, newim in zip(frames, processed):
        #resizing images for display
        #cv2.imshow('skrrt',cv2.resize(im,(int(im.shape[0]/5),int(im.shape[1]/5))))
        cv2.waitKey(0)
        cv2.imshow('skrrt',cv2.resize(newim,(int(newim.shape[0]/5),int(newim.shape[1]/5))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
