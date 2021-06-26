import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single


# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
img_path = filedialog.askopenfilename()
# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh

Dmax = 288
Dmin = 124

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)
# Tách biển số ra khỏi xe
_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


# Cau hinh tham so mặc định cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu


model_svm = cv2.ml.SVM_load('svm.xml')



if (len(LpImg)):

    plate_info = ""

    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

    roi = LpImg[0]

    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)


    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]
    heightTB = round(binary.shape[0] / 2)
    weightTB = round(binary.shape[0]*10 )
    binary1 = binary[0:heightTB, 0:weightTB ]
    binary2 = binary[heightTB:binary.shape[0], :]
    cv2.imshow("Anh bien so sau threshold", binary)
    cv2.waitKey(delay=30)


    binary = binary1
  # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
            if h / roi.shape[0] >= 0.3: # chon cac contour cao tu 30% bien so tro len bien 2 dong
                # Ve khung chu nhat quanh so
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0,0, 255), 2)

                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                curr_num = np.array(curr_num, dtype=np.float32)
                curr_num = curr_num.reshape(-1, digit_w * digit_h)

                # Dua vao model SVM
                result = model_svm.predict(curr_num)[1]
                result = int(result[0, 0])

                if result<=9: # Neu la so thi hien thi luon
                    result = str(result)
                else: #Neu la chu thi chuyen bang ASCII
                    result = chr(result)

                plate_info +=result


    binary = binary2
    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    plate_info_1 = ""
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 1.5 <= ratio <= 3.5:  # Chon cac contour dam bao ve ratio w/h
            if h / roi.shape[0] >= 0.3:  # chon cac contour cao tu 30% bien so tro len bien 2 dong
                # Ve khung chu nhat quanh so
                cv2.rectangle(roi, (x, y+100), (x + w, y+15 + h*2), (0, 255, 255), 2)

                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                curr_num = np.array(curr_num, dtype=np.float32)
                curr_num = curr_num.reshape(-1, digit_w * digit_h)

                # Dua vao model SVM
                result = model_svm.predict(curr_num)[1]
                result = int(result[0, 0])

                if result <= 9:  # Neu la so thi hien thi luon
                    result = str(result)
                else:  # Neu la chu thi chuyen bang ASCII
                    result = chr(result)

                plate_info_1 += result
    
    cv2.imshow("Cac contour tim duoc", roi)
    cv2.waitKey(delay=30)
    # Viet bien so len anh
    
    # cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), lineType=cv2.LINE_AA)

    cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 244, 255), lineType=cv2.LINE_AA)
    cv2.putText(Ivehicle,fine_tune(plate_info_1),(50, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 244, 255), lineType=cv2.LINE_AA)

    # Hien thi anh
    print("Bien so=", plate_info +" " + plate_info_1)
    cv2.imshow("Hinh anh output",Ivehicle)
    cv2.waitKey()
    cv2.destroyAllWindows()
