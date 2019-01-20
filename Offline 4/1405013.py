import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt


def distance(ref_image, frame_image, M, N, m, n):
    ref_image = np.array(ref_image)
    frame_image = np.array(frame_image)
    arr1 = frame_image[m: m + M, n: n + N].astype(np.int64)
    arr2 = ref_image.astype(np.int64)
    arr3 = np.absolute(arr2 - arr1)
    arr4 = arr3 * arr3
    return np.sum(arr4)


def exhaustive_search(ref_img, frames, frame_rate, p):
    cnt = 0
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
        d_min = math.inf
        d_min_i = d_min_j = -1
        if k == 0:
            for i in range(I - M):
                for j in range(J - N):
                    d = distance(ref_img, frame_img, M, N, i, j)
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        d_min_i = i
                        d_min_j = j
            previous_match_y = d_min_i
            previous_match_x = d_min_j
        else:
            for i in range(previous_match_y - p, previous_match_y + p + 1):
                for j in range(previous_match_x - p, previous_match_x + p + 1):
                    if i < 0 or j < 0 or i >= I - M or j >= J - N:
                        continue
                    d = distance(ref_img, frame_img, M, N, i, j)
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        d_min_i = i
                        d_min_j = j
            previous_match_y = d_min_i
            previous_match_x = d_min_j

        rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)

        for i in range(d_min_i - 1, d_min_i + M + 1):
            rgb_frame[i][d_min_j - 1][0] = np.int8(0)
            rgb_frame[i][d_min_j - 1][1] = np.int8(0)
            rgb_frame[i][d_min_j - 1][2] = np.int8(255)

            rgb_frame[i][d_min_j + N][0] = np.int8(0)
            rgb_frame[i][d_min_j + N][1] = np.int8(0)
            rgb_frame[i][d_min_j + N][2] = np.int8(255)
        for j in range(d_min_j - 1, d_min_j + N + 1):
            rgb_frame[d_min_i - 1][j][0] = np.int8(0)
            rgb_frame[d_min_i - 1][j][1] = np.int8(0)
            rgb_frame[d_min_i - 1][j][2] = np.int8(255)

            rgb_frame[d_min_i + M][j][0] = np.int8(0)
            rgb_frame[d_min_i + M][j][1] = np.int8(0)
            rgb_frame[d_min_i + M][j][2] = np.int8(255)

        output_frames.append(rgb_frame)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('Output Videos/out_exhaust_' + str(p) + '.avi', fourcc, frame_rate, (J, I), True)

    for output_frame in output_frames:
        out.write(output_frame)
    out.release()

    print("Output video saved for exhaustive search")
    return (cnt*1.0)/len(frames)


def logarithmic_search(ref_img, frames, frame_rate, p):
    cnt = 0
    M = ref_img.shape[0]
    N = ref_img.shape[1]
    print("Reference image size " + str((N, M)))

    I = frames[0].shape[0]
    J = frames[0].shape[1]
    print("Video frame size " + str((J, I)))

    output_frames = []
    previous_match_x = previous_match_y = 0
    print("Total frames " + str(len(frames)))
    d_min_i = d_min_j = -1
    for k in range(len(frames)):
        frame_img = frames[k]
        if k == 0:
            d_min = math.inf
            d_min_i = d_min_j = -1
            for i in range(I - M):
                for j in range(J - N):
                    d = distance(ref_img, frame_img, M, N, i, j)
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        d_min_i = i
                        d_min_j = j
            previous_match_y = d_min_i
            previous_match_x = d_min_j
        else:
            p_tmp = p
            spacing = math.pow(2, math.ceil(math.log(p, 2)) - 1)
            while spacing >= 1:
                d_min = math.inf
                d_min_i = d_min_j = -1
                for i in [previous_match_y - p_tmp, previous_match_y, previous_match_y + p_tmp]:
                    for j in [previous_match_x - p_tmp, previous_match_x, previous_match_x + p_tmp]:
                        if i < 0 or j < 0 or i >= I - M or j >= J - N:
                            continue
                        d = distance(ref_img, frame_img, M, N, i, j)
                        cnt += 1
                        if d < d_min:
                            d_min = d
                            d_min_i = i
                            d_min_j = j
                previous_match_y = d_min_i
                previous_match_x = d_min_j
                p_tmp = round(p_tmp / 2.0)
                spacing /= 2

        rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)

        for i in range(d_min_i - 1, d_min_i + M + 1):
            rgb_frame[i][d_min_j - 1][0] = np.int8(0)
            rgb_frame[i][d_min_j - 1][1] = np.int8(0)
            rgb_frame[i][d_min_j - 1][2] = np.int8(255)

            rgb_frame[i][d_min_j + N][0] = np.int8(0)
            rgb_frame[i][d_min_j + N][1] = np.int8(0)
            rgb_frame[i][d_min_j + N][2] = np.int8(255)
        for j in range(d_min_j - 1, d_min_j + N + 1):
            rgb_frame[d_min_i - 1][j][0] = np.int8(0)
            rgb_frame[d_min_i - 1][j][1] = np.int8(0)
            rgb_frame[d_min_i - 1][j][2] = np.int8(255)

            rgb_frame[d_min_i + M][j][0] = np.int8(0)
            rgb_frame[d_min_i + M][j][1] = np.int8(0)
            rgb_frame[d_min_i + M][j][2] = np.int8(255)

        output_frames.append(rgb_frame)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('Output Videos/out_log_' + str(p) + '.avi', fourcc, frame_rate, (J, I), True)

    for output_frame in output_frames:
        out.write(output_frame)
    out.release()

    print("Output video saved for 2D logarithmic search")
    return (cnt * 1.0) / len(frames)


def hierarchical_search(ref_img, frames, frame_rate, p):
    cnt = 0
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
        if k == 0:
            d_min = math.inf
            d_min_i = d_min_j = -1
            for i in range(I - M):
                for j in range(J - N):
                    cnt += 1
                    d = distance(ref_img, frame_img, M, N, i, j)
                    if d < d_min:
                        d_min = d
                        d_min_i = i
                        d_min_j = j
            previous_match_y = d_min_i
            previous_match_x = d_min_j
        else:
            lvl_refs = [ref_img]
            lvl_frames = [frame_img]
            lvl_refs.append(cv2.pyrDown(lvl_refs[0]))
            lvl_frames.append(cv2.pyrDown(lvl_frames[0]))
            lvl_refs.append(cv2.pyrDown(lvl_refs[1]))
            lvl_frames.append(cv2.pyrDown(lvl_frames[1]))

            d_min = math.inf
            d_min_i = d_min_j = -1
            p_tmp = round(p/4.0)
            center_x = math.floor(previous_match_x/4.0)
            center_y = math.floor(previous_match_y/4.0)
            for i in range(center_y - p_tmp, center_y + p_tmp + 1):
                for j in range(center_x - p_tmp, center_x + p_tmp + 1):
                    I_ = lvl_frames[2].shape[0]
                    J_ = lvl_frames[2].shape[1]
                    M_ = lvl_refs[2].shape[0]
                    N_ = lvl_refs[2].shape[1]
                    if i < 0 or j < 0 or i >= I_ - M_ or j >= J_ - N_:
                        continue
                    d = distance(lvl_refs[2], lvl_frames[2], lvl_refs[2].shape[0], lvl_refs[2].shape[1], i, j)
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        d_min_i = i
                        d_min_j = j
            y1 = d_min_i
            x1 = d_min_j

            d_min = math.inf
            p_tmp = 1
            center_x = 2*x1
            center_y = 2*y1
            for i in range(center_y - p_tmp, center_y + p_tmp + 1):
                for j in range(center_x - p_tmp, center_x + p_tmp + 1):
                    I_ = lvl_frames[1].shape[0]
                    J_ = lvl_frames[1].shape[1]
                    M_ = lvl_refs[1].shape[0]
                    N_ = lvl_refs[1].shape[1]
                    if i < 0 or j < 0 or i >= I_ - M_ or j >= J_ - N_:
                        continue
                    d = distance(lvl_refs[1], lvl_frames[1], lvl_refs[1].shape[0], lvl_refs[1].shape[1], i, j)
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        d_min_i = i
                        d_min_j = j
            y2 = d_min_i
            x2 = d_min_j

            d_min = math.inf
            p_tmp = 1
            center_x = 2*x2
            center_y = 2*y2
            for i in range(center_y - p_tmp, center_y + p_tmp + 1):
                for j in range(center_x - p_tmp, center_x + p_tmp + 1):
                    I_ = lvl_frames[0].shape[0]
                    J_ = lvl_frames[0].shape[1]
                    M_ = lvl_refs[0].shape[0]
                    N_ = lvl_refs[0].shape[1]
                    if i < 0 or j < 0 or i >= I_ - M_ or j >= J_ - N_:
                        continue
                    d = distance(lvl_refs[0], lvl_frames[0], lvl_refs[0].shape[0], lvl_refs[0].shape[1], i, j)
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        d_min_i = i
                        d_min_j = j
            previous_match_y = d_min_i
            previous_match_x = d_min_j

        rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)

        for i in range(d_min_i - 1, d_min_i + M):
            rgb_frame[i][d_min_j - 1][0] = np.int8(0)
            rgb_frame[i][d_min_j - 1][1] = np.int8(0)
            rgb_frame[i][d_min_j - 1][2] = np.int8(255)

            rgb_frame[i][d_min_j + N][0] = np.int8(0)
            rgb_frame[i][d_min_j + N][1] = np.int8(0)
            rgb_frame[i][d_min_j + N][2] = np.int8(255)
        for j in range(d_min_j - 1, d_min_j + N):
            rgb_frame[d_min_i - 1][j][0] = np.int8(0)
            rgb_frame[d_min_i - 1][j][1] = np.int8(0)
            rgb_frame[d_min_i - 1][j][2] = np.int8(255)

            rgb_frame[d_min_i + M][j][0] = np.int8(0)
            rgb_frame[d_min_i + M][j][1] = np.int8(0)
            rgb_frame[d_min_i + M][j][2] = np.int8(255)
        output_frames.append(rgb_frame)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('Output Videos/out_hierarchy_' + str(p) + '.avi', fourcc, frame_rate, (J, I), True)

    for output_frame in output_frames:
        out.write(output_frame)
    out.release()

    print("Output video saved for hierarchical search")
    return (cnt * 1.0) / len(frames)


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

ex_counts = []
log_counts = []
hie_counts = []

p_list = range(5, 51, 5)

for p_ in p_list:
    start = time.time()
    ex_counts.append(exhaustive_search(grayImg, frame_array, fps, p_))
    end = time.time()
    print("Exhaustive search took " + str((end - start) / 60) + " minutes to run for p = " + str(p_))

    start = time.time()
    log_counts.append(logarithmic_search(grayImg, frame_array, fps, p_))
    end = time.time()
    print("2D Logarithmic search took " + str((end - start) / 60) + " minutes to run for p = " + str(p_))

    start = time.time()
    hie_counts.append(hierarchical_search(grayImg, frame_array, fps, p_))
    end = time.time()
    print("Hierarchical search took " + str((end - start) / 60) + " minutes to run for p = " + str(p_))

plt.plot(p_list, ex_counts, label='Exhaustive Search')
plt.xlabel('p')
plt.ylabel('Frame search count')
plt.show()

plt.plot(p_list, log_counts, label='2D Logarithmic Search')
plt.xlabel('p')
plt.ylabel('Frame search count')
plt.show()

plt.plot(p_list, hie_counts, label='Hierarchical Search')
plt.xlabel('p')
plt.ylabel('Frame search count')
plt.show()
