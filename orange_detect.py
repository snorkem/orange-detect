#!/usr/bin/env python3
import cv2
import numpy as np
import fire
from itertools import groupby, count
from operator import itemgetter
from matplotlib import pyplot as plt

# Output path — Change this to your location!
file_out = open(r'/Users/blb_ae/Desktop/frame_flags.txt', 'w')


def write_out(flag_count):
    new_list = []
    for k, g in groupby(enumerate(flag_count), lambda x: x[0]-x[1]):
        new_list.append(list(map(itemgetter(1), g)))
    # print(new_list, sub_list)
    for i in range(len(new_list)):
        print('Writing frame range: ' + str(new_list[i][0]) + ' to ' + str(new_list[i][-1]))
        file_out.write('Frame range: ' + str(new_list[i][0]) + ' to ' + str(new_list[i][-1]) + '\n')

def plot_orange_freq(pix_counter, frame_num):
    data = pix_counter
    #print(data)
    frame_num = frame_num
    #fig, ax = plt.subplots()
    #ax = fig.add_subplot(111)
    #plt.setp(ax.)
    x_axe = [i for i in range(len(frame_num))]
    # plt.axis([0,(frame_num[-1]),0,20000])
    plt.scatter(frame_num, data, c='r')
    #plt.ylim(0, 10000)
    #plt.yticks([int(data[0]), int(data[-1])])
    #print(int(min(data)))
    #plt.ylim(0, int(max(data)))
    #plt.ylim(int(max(frame_num)))
    ticks = np.arange(0,101,100)
    plt.yticks(ticks)
    #plt.yticks([int(min(data)), int(max(data))])
    plt.ylabel('# of pixels')
    plt.xlabel('Frame  #')
    plt.title('Orange Pixel Graph')
    plt.show()

def apply_filters(frame, threshold):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_orange = np.array([10, 80, 50])
    high_orange = np.array([20, 255, 255])
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange = cv2.bitwise_and(frame, frame, mask=orange_mask)
    bi_blur = cv2.bilateralFilter(orange, 15, 75, 75)
    # get a one dimensional Gaussian Kernel
    gaussian_kernel_x = cv2.getGaussianKernel(4, 4)
    gaussian_kernel_y = cv2.getGaussianKernel(4, 4)
    # # converting to two dimensional kernel using matrix multiplication
    gaussian_kernel = gaussian_kernel_x * gaussian_kernel_y.T
    # filtered_image = cv2.filter2D(bi_blur, -1, gaussian_kernel) # The blur could be done in 1 line, if revised.
    opening = cv2.morphologyEx(bi_blur, cv2.MORPH_OPEN, gaussian_kernel)
    ret, th1 = cv2.threshold(bi_blur, threshold, 255, cv2.THRESH_BINARY)
    # Final filtered result
    filtered = cv2.cvtColor(th1, cv2.COLOR_BGR2GRAY)
    return filtered

# default values declared in the function call
def main(vid, sensitivity=50, threshold=150, preview=True, graph=False):
    """
    Simple image processor using OpenCV to detect and flag frames with the "safety orange" color present. Designed
    for finding the "black box" in survey drone footage of an airplane crash. \n
    Author: Alex Fichera
    Version: 0.5
    :param vid: path to video file
    :param sensitivity: Number of pixels in frame that will trigger a flag. Value can be 1 to whatever. Default is 50.
    :param threshold: Filter threshold to be considered as 'orange'. Value can be 0-255. Default is 150.
    :param preview: Show preview. True or False. Defaults to True.
    :param graph: Show graph of orange pixels for frames in which they are detected. True or False. Defaults to False.

    Example Usage:
    orange_detect.py <file_path> --sensitivity <value> --threshold <value> --preview <False> --graph <True>

    Press 'q' to exit this help page.
    """
    stream = cv2.VideoCapture(vid)
    # stream.set(cv2.CAP_PROP_FPS, 50)
    flag_count = []
    pix_counter = []
    print('Pixel sensitivity is: ' + str(sensitivity))
    print('Threshold is: ' + str(threshold))
    if sensitivity is None:
        print('its none')
    while stream.isOpened():
        ret, frame = stream.read()
        if frame is None:
            break
        # frame = np.array(frame, dtype=np.uint8) ---- This was to fix a bug that didn't really exist.

        cur_frame = int(stream.get(cv2.CAP_PROP_POS_FRAMES))
        prev_orange_frame = 0
        if len(flag_count): # This was when I wanted to show cur and prev frames.
            prev_orange_frame = int(flag_count[-1])

        orange_filter = apply_filters(frame, threshold)
        n_orange_pix = str(cv2.countNonZero(orange_filter))
        # Opening the window to show the processing...
        if preview is True:
            cv2.imshow("Alex's Black Box Detector (Beta)", cv2.resize(orange_filter, (1280, 720)))
        print('Processing Frame: ' + str(int(stream.get(cv2.CAP_PROP_POS_FRAMES))) + ' of ' +
              str(int(stream.get(cv2.CAP_PROP_FRAME_COUNT))) + '\n'
              + 'Number of Orange Pixels Detected: ' + n_orange_pix)
        if cv2.countNonZero(orange_filter) > sensitivity:
            flag_count.append(cur_frame)
            pix_counter.append(n_orange_pix)
            print('Orange flagged in frame #: ' + str(cur_frame) + '\n' + '##################')
            print('flag count is: ' + str(len(flag_count)))
        else:
            print('Nothing to flag...' + '\n' + '##################')
        key = cv2.waitKey(1)
        if key == 27:
            break
    stream.release()
    cv2.destroyWindow("Alex's Black Box Detector (Beta")
    write_out(flag_count)
    if graph is True:
        plot_orange_freq(pix_counter, flag_count)
    else:
        quit(0)

if __name__ == '__main__':
  fire.Fire(main)