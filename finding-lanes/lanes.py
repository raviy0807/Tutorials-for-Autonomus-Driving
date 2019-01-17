#Library
import cv2
import numpy as np

def canny_fn(lane_image):
    gray_image = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int((3/5)*y1)
    x1 = int((y1-intercept)/slope)
    x2 = int( (y2-intercept)/ slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    return np.array([left_line,right_line])


#Read Image
'''image = cv2.imread('c:/Users/Ravi/Documents/GitHub/Udemy-Complete-Self-Driving-Course/finding-lanes/test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
roi = region_of_interest(canny)
lines =  cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
average_line = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, average_line)
combo_image = cv2.addWeighted(lane_image,0.7, line_image, 1, 1)
cv2.imshow('result',combo_image)
cv2.waitKey(0)'''


#Capture Lane in Video
cap = cv2.VideoCapture("C:/Users/Ravi/Documents/GitHub/Udemy-Complete-Self-Driving-Course/finding-lanes/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny = canny_fn(frame)
    roi = region_of_interest(canny)
    lines =  cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_line = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_line)
    combo_image = cv2.addWeighted(frame,0.7, line_image, 1, 1)
    cv2.imshow('result',combo_image)
    cv2.waitKey(1)
    print("Inside while!!")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindow()
