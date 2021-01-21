import cv2
import numpy as np
import dlib
from scipy import signal
from scipy.signal import butter, lfilter
from scipy import fftpack
import matplotlib.pyplot as plt


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "drive/MyDrive/BTP - rakshak/shape_predictor_81_face_landmarks.dat")


def capturvideo(videoname):
    vid = cv2.VideoCapture(videoname)
    framecnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frameht = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framewd = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    video = np.zeros((framecnt, 512, 512), float)
    x = 0
    #video = []
    face_rects = ()
    x1 = 0
    x2 = 0
    x3 = 0
    y1 = 0
    y2 = 0
    y6 = 0
    y7 = 0
    while vid.isOpened():
        ret, img = vid.read()
        if ret:
            green = img[:, :, 1]
            if x == 0:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                faces = detector(gray)
                print(len(faces))
                for face in faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()
                    landmarks = predictor(img, face)
                    x3 = landmarks.part(17).x
                    y3 = landmarks.part(17).y
                    x4 = landmarks.part(19).x
                    y4 = landmarks.part(19).y
                    x5 = landmarks.part(26).x
                    y5 = landmarks.part(26).y
                    x6 = landmarks.part(70).x
                    y6 = landmarks.part(70).y
                    x7 = landmarks.part(21).x
                    y7 = landmarks.part(21).y
                    break
            #img1 = cv2.rectangle(img, (x3, y6), (x2, y7), (0, 0, 255), 3)
            img1 = green[y6:y7, x3:x2]
            # print(x)
            # cv2.imwrite(str(x)+'.jpg',img1)
            gface1 = cv2.resize(img1, (512, 512))
            video[x] = gface1
            x = x+1

        else:
            break
    vid.release()
    return video, fps, framecnt


def gaussian_pyramid(img):
    g = img.copy()
    gpA = [g]
    for i in range(3):
        g = cv2.pyrDown(g)
        gpA.append(g)
    return gpA


def laplacian_pyramid(img):
    gpA = gaussian_pyramid(img)
    lpA = [gpA[-1]]
    # print(img.shape[:])
    for i in range(2, 0, -1):
        a = cv2.pyrUp(gpA[i])
        #print(len(gpA[i-1]), len(a))
        c = cv2.subtract(gpA[i-1], a)
        # print(len(c))
        lpA.append(a)
    return lpA


def laplacian_video(frames):
    lap_video = []

    for i, frame in enumerate(frames):
        pyramid = laplacian_pyramid(frame)
        for j in range(3):
            if i == 0:
                lap_video.append(
                    np.zeros((len(frames), pyramid[j].shape[0], pyramid[j].shape[1])))
            lap_video[j][i] = pyramid[j]

    return lap_video


def ideal_bandpass(low, high, img):
    ans = []
    for x in img:
        g = x.copy()
        row, cols = x.shape[:2]
        for i in range(row):
            for j in range(cols):
                if abs(x[i, j]) > high or abs(x[i, j] < low):
                    g[i, j] = 0
        ans.append(g)
    return ans


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y


vd, fps, framecnt = capturvideo('drive/MyDrive/BTP - rakshak/008.mp4')
high = 4
low = 0.8
FFT_vid = []
IFFT_vid = []
kd = laplacian_video(vd)

for i, video in enumerate(kd):
    if i == 0 or i == len(kd)-1:
        continue

    f = fftpack.fft(video, axis=0)
    # print(f.shape[:])
    # plt.subplot(151),plt.imshow(np.log(1+np.abs(f)),"gray"),plt.title("fft")
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0/fps)
    # print(f)
    # print(frequencies)
    # bound_low = (np.abs(frequencies-low)).argmin()
    # bound_high = (np.abs(frequencies-high)).argmin()
    # f[:bound_low] = 0
    # f[bound_high:-bound_high] =0
    # f[-bound_low:] = 0
    # plt.subplot(152),plt.imshow(np.log(1+np.abs(f)),"gray"),plt.title("fft2")
    f = butter_bandpass_filter(f, low, high, fps)
    # plt.show()
    fft_maximus = []

    for j in range(f.shape[0]):
        if low <= frequencies[j] <= high:
            fftMap = abs(f[j])
            fft_maximus.append(fftMap.max())
        else:
            fft_maximus.append(0)

    peaks, properties = signal.find_peaks(fft_maximus)
    max_peak = -1
    max_freq = 0

    for peak in peaks:
        if fft_maximus[peak] > max_freq:
            max_freq = fft_maximus[peak]
            max_peak = peak

    print(frequencies[max_peak] * 60)
    iff = fftpack.ifft(f, axis=0)
    iff = np.abs(iff)
    iff *= 100
    kd[i] += iff
    # print(find_heart_rate(f,frequencies,low,high))

final = []

for i in range(framecnt):
    prev_frame = kd[-1][i]
    for l in range(len(kd)-1, 0, -1):
        pyr_up_frame = cv2.pyrUp(prev_frame)
        (height, width) = pyr_up_frame.shape
        prev_level_frame = kd[l-1][i]
        prev_level_frame = cv2.resize(prev_level_frame, (height, width))
        prev_frame = pyr_up_frame + prev_level_frame

    min_val = min(0.0, prev_frame.min())
    prev_frame = prev_frame + min_val
    max_val = max(1.0, prev_frame.max())
    prev_frame = prev_frame / max_val
    prev_frame = prev_frame * 255

    prev_frame = cv2.convertScaleAbs(prev_frame)
    final.append(prev_frame)

#cv2.imwrite("afteradding.jpg", final[0])

height, width = final[0].shape[0:2]
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height), 1)

for i in range(len(final)):
    fr = final[i]
    fr = np.uint8(fr)
    out.write(fr)
out.release()
