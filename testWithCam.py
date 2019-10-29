from train import *
import cv2
import glob
import time
import requests
WEIGHTS_FILE = "ruler.h5"
THRESHOLD = 0.99
EPSILON = 0.08

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)
    cap = cv2.VideoCapture(0)
    x0 = 1
    x1 = 1
    y0 = 1
    y1 = 1
    while True:
        ret, frame = cap.read()
        #cv2.imwrite("output.png", frame)
        unscaled = frame
        image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
        feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

        region = np.squeeze(model.predict(feat_scaled[np.newaxis,:]))

        output = np.zeros(region.shape, dtype=np.uint8)
        output[region > THRESHOLD] = 1

        __,contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contours) > 0):
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, EPSILON * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            x0 = np.rint(x * unscaled.shape[1] / output.shape[1]).astype(int)
            x1 = np.rint((x + w) * unscaled.shape[1] / output.shape[1]).astype(int)
            y0 = np.rint(y * unscaled.shape[0] / output.shape[0]).astype(int)
            y1 = np.rint((y + h) * unscaled.shape[0] / output.shape[0]).astype(int)
            cv2.rectangle(unscaled, (x0, y0), (x1, y1), (0, 255, 0), 3)
            
            print("Ruler size -> Width:", x1-x0, "Height:", y1-y0)

        imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower=np.array([35,30,5])
        upper=np.array([70,255,250])
        mask = cv2.inRange(imggray, lower, upper)
        ___, contours, hierarchy =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        C=max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(C)#計算邊界框座標
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        rect = cv2.minAreaRect(C)#計算包圍目標的最小矩形區域
        box = cv2.boxPoints(rect) #計算最小矩形的座標
        box = np.int0(box) #座標變為整數
        cv2.drawContours(unscaled, [box], 0, (0,0,255),1)
        print("Object size -> Width:", w, "Height:", h)
        if x1-x0 != 0 and y1-y0 != 0:
            #print("Real object = ", 5*w/(x1-x0), "cm", 5*h/(y1-y0), "cm")
            font = ""
            cv2.putText(unscaled, str(5*h/(y1-y0))+" cm", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255,0), 1, cv2.LINE_AA)
            print("Real object height = ", 5*h/(y1-y0), "cm")
        #files = {"upload_file":open("output.png", "rb")}
        #data = {"height":5*h/(y1-y0)}
        #s = requests.post("https://kaibao.tzml-lab.tw/api/uploadImg", files=files, data=data, verify=False)
        cv2.imshow("image", unscaled)
        #time.sleep(10)
        #break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
