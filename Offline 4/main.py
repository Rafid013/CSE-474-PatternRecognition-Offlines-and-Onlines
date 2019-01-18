import cv2
import numpy as np
import math
import time


def exhaustive_distance(ref_image, frame_image, M, N, m, n):
    ref_image = np.array(ref_image)
    frame_image = np.array(frame_image)
    arr1 = frame_image[m: m + M, n: n + N].astype(np.int64)
    arr2 = ref_image.astype(np.int64)
    c1 = np.tensordot(arr1, arr2)
    c2 = np.tensordot(arr1, arr1)
    c3 = np.tensordot(arr2, arr2)
    c = c1/math.sqrt(c2*c3)
    return c


def exhaustive_search(ref_img, frames, frame_rate, p):
    M = ref_img.shape[0]
    N = ref_img.shape[1]
    print("Reference image size " + str((N, M)))

    I = frames[0].shape[0]
    J = frames[0].shape[1]
    print("Video frame size " + str((J, I)))

    output_frames = []
    previous_match_x = previous_match_y = 0
    print("Total frames " + str(len(frames)))
    for k in range(len(frames)):
        frame_img = frames[k]
        c_min = math.inf
        c_min_i = c_min_j = -1
        if k == 0:
            for i in range(I - M):
                for j in range(J - N):
                    d = exhaustive_distance(ref_img, frame_img, M, N, i, j)
                    if d < c_min:
                        c_min = d
                        c_min_i = i
                        c_min_j = j
            previous_match_x = c_min_i
            previous_match_y = c_min_j
        else:
            for i in range(previous_match_x - p, previous_match_x + p + 1):
                if i < 0:
                    i = 0
                if i >= I - M:
                    i = I - M - 1
                for j in range(previous_match_y - p, previous_match_y + p + 1):
                    if j < 0:
                        j = 0
                    if j >= J - N:
                        j = j - N - 1
                    d = exhaustive_distance(ref_img, frame_img, M, N, i, j)
                    if d < c_min:
                        c_min = d
                        c_min_i = i
                        c_min_j = j
            previous_match_x = c_min_i
            previous_match_y = c_min_j

        rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)

        for i in range(c_min_i - 5, c_min_i + M - 4):
            rgb_frame[i][c_min_j + 3][0] = np.int8(0)
            rgb_frame[i][c_min_j + 3][1] = np.int8(0)
            rgb_frame[i][c_min_j + 3][2] = np.int8(255)

            rgb_frame[i][c_min_j + N + 1][0] = np.int8(0)
            rgb_frame[i][c_min_j + N + 1][1] = np.int8(0)
            rgb_frame[i][c_min_j + N + 1][2] = np.int8(255)
        for j in range(c_min_j + 3, c_min_j + N + 2):
            rgb_frame[c_min_i - 5][j][0] = np.int8(0)
            rgb_frame[c_min_i - 5][j][1] = np.int8(0)
            rgb_frame[c_min_i - 5][j][2] = np.int8(255)

            rgb_frame[c_min_i + M - 5][j][0] = np.int8(0)
            rgb_frame[c_min_i + M - 5][j][1] = np.int8(0)
            rgb_frame[c_min_i + M - 5][j][2] = np.int8(255)

        output_frames.append(rgb_frame)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('out_exhaust.avi', fourcc, frame_rate, (J, I), True)

    for output_frame in output_frames:
        out.write(output_frame)
    out.release()

    print("Output video saved for exhaustive search")


cap = cv2.VideoCapture('movie.mov')

count = 0

frame_array = []

fps = cap.get(cv2.CAP_PROP_FPS)

# read all the frames of the test video
while True:
    # cv2.imwrite("TEMP/frame%d.jpg" % count, frame)  # save frame as JPEG file
    ret, frame = cap.read()
    if not ret:
        break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_array.append(grayFrame)
    # print('Read a new frame: ', ret)
    count += 1

cap.release()

# read the reference image
img = cv2.imread('reference.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

start = time.time()
exhaustive_search(grayImg, frame_array, fps, 20)
end = time.time()

print("Took " + str((end - start)/60) + " minutes to run")
