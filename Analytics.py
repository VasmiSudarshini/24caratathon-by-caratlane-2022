import cv2
import numpy as np
import tensorflow as tf
import time

dict_s ={}
dict_corr = {1: [0.352865, 0.789931, 0.025521, 0.024306],
                     2: [0.382812, 0.782292, 0.023958, 0.025694],
                     3: [0.414062, 0.777431, 0.025000, 0.028472],
                     4: [0.442448, 0.771181, 0.022396, 0.029861],
                     5: [0.474740, 0.764583, 0.023438, 0.025000],
                     6: [0.355208, 0.815972, 0.022917, 0.023611],
                     7: [0.383333, 0.811111, 0.021875, 0.026389],
                     8: [0.414583, 0.806597, 0.025000, 0.027083],
                     9: [0.444792, 0.799653, 0.023958, 0.025694],
                     10: [0.478125, 0.793403, 0.023958, 0.028472],
                     11: [0.356771, 0.851042, 0.023958, 0.029861],
                     12: [0.391406, 0.843403, 0.023438, 0.027083],
                     13: [0.421354, 0.837153, 0.022917, 0.025694],
                     14: [0.448698, 0.828819, 0.024479, 0.027083],
                     15: [0.482031, 0.825694, 0.023438, 0.027778],
                     16: [0.400521, 0.901042, 0.023958, 0.028472],
                     17: [0.439844, 0.893403, 0.026562, 0.031250],
                     18: [0.494271, 0.882986, 0.025000 ,0.032639],
                     19: [0.403125, 0.934375, 0.023958, 0.031250],
                     20: [0.445052, 0.925694, 0.024479, 0.030556],
                     21: [0.497396, 0.918403, 0.023958, 0.032639],
                     22: [0.407813, 0.970833, 0.026042, 0.033333],
                     23: [0.449479, 0.964931, 0.025000, 0.032639],
                     24: [0.505990, 0.952083 ,0.025521, 0.030556]}
class img_to_num:
    def __init__(self):
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.cls = ['True', 'False']
        self.model = tf.keras.models.load_model(r"keras_model.h5", compile=False)


    def findId(self, img):
        image = cv2.resize(img, (224, 224))
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        
        prediction = self.model.predict_on_batch(self.data)
        return self.cls[np.argmax(prediction[0], 0)], max(prediction[0])

        
np.set_printoptions(suppress=True)
predict = img_to_num()


vid = cv2.VideoCapture(r"video.mp4")

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while(True):
    new_frame_time = time.time()
    ret, frame = vid.read()
    ret, frame = vid.read()
    ret, frame = vid.read()
    ret, frame = vid.read()
    ret, frame = vid.read()
    ret, frame = vid.read()
    
    dh, dw, c = frame.shape
    areas = dict_corr.items()
    for j,i in areas:
        # print(i)
        
        x_center = round(i[0] * dw)
        y_center = round(i[1] * dh)
        w = round(i[2] * dw)
        h = round(i[3] * dh)
        x = round(x_center - w / 2)
        y = round(y_center - h / 2)
        imgCrop = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y),(x+w, y + h), (255, 0, 0), 2)
        overlay = frame.copy()
        new_out = frame.copy()
	# draw a red rectangle surrounding Adrian in the image
	# along with the text "PyImageSearch" at the top-left
	# corner
        cv2.rectangle(overlay, (20, 10), (800, 760),(0, 0, 0), -1)
        cv2.rectangle(overlay, (1150, 20), (1900, 460), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.09, new_out, (1 - 0.09),0, frame)
        # cFram = frame[i[1]:i[3], i[0]:i[2]]
        start = time.time()
        area_status = predict.findId(imgCrop)
        # print(j, ">", area_status)
        area_status_new = area_status[0]
        dict_new = {j: area_status_new}
        dict_s.update(dict_new)
        # status = str(j)+"-"+ area_status_new
        # cv2.putText(frame, status, (50, 250),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        # print(j, ">", area_status_new)
    
    if dict_s[1] == "True" :
        cv2.putText(frame, 'Rack 1 First Row Product_1 - Availble',(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[1] == "False":
        cv2.putText(frame, 'Rack 1 First Row Product_1 - Not Availble', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    
    if dict_s[2] == "True":
        cv2.putText(frame, 'Rack 1 First Row Product_2 - Availble ',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[2] == "False":
        cv2.putText(frame, 'Rack 1 First Row Product_2 - Not Availble', (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[3] == "True":
        cv2.putText(frame, 'Rack 1 First Row Product_3 - Availble ', (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[3] == "False":
        cv2.putText(frame, 'Rack 1 First Row Product_3 - Not Availble ', (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[4] == "True":
        cv2.putText(frame, 'Rack 1 First Row Product_4 - Availble ', (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[4] == "False":
        cv2.putText(frame, 'Rack 1 First Row Product_4 - Not Availble ', (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[5] == "True":
        cv2.putText(frame, 'Rack 1 First Row Product_5 - Availble ', (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[5] == "False":
        cv2.putText(frame, 'Rack 1 First Row Product_5 - Not Availble ', (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[6] == "True":
        cv2.putText(frame, 'Rack 1 Second Row Product_1 - Availble ', (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[6] == "False":
        cv2.putText(frame, 'Rack 1 Second Row Product_1 - Not Availble ', (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[7] == "True":
        cv2.putText(frame, 'Rack 1 Second Row Product_2 - Availble ', (50, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[7] == "False":
        cv2.putText(frame, 'Rack 1 Second Row Product_2 - Not Availble ', (50, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[8] == "True":
        cv2.putText(frame, 'Rack 1 Second Row Product_3 - Availble ', (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[8] == "False":
        cv2.putText(frame, 'Rack 1 Second Row Product_3 - Not Availble ', (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    
    if dict_s[9] == "True":
        cv2.putText(frame, 'Rack 1 Second Row Product_4 - Availble ', (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[9] == "False":
        cv2.putText(frame, 'Rack 1 Second Row Product_4 - Not Availble ', (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[10] == "True":
        cv2.putText(frame, 'Rack 1 Second Row Product_5 - Availble ', (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[10] == "False":
        cv2.putText(frame, 'Rack 1 Second Row Product_5 - Not Availble ', (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[11] == "True":
        cv2.putText(frame, 'Rack 1 Third Row Product_1 - Availble ', (50, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[11] == "False":
        cv2.putText(frame, 'Rack 1 Third Row Product_1 - Not Availble ', (50, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[12] == "True":
        cv2.putText(frame, 'Rack 1 Third Row Product_2 - Availble ', (50, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[12] == "False":
        cv2.putText(frame, 'Rack 1 Third Row Product_2 - Not Availble ', (50, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[13] == "True":
        cv2.putText(frame, 'Rack 1 Third Row Product_3 - Availble ', (50, 650),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[13] == "False":
        cv2.putText(frame, 'Rack 1 Third Row Product_3 - Not Availble ', (50, 650),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[14] == "True":
        cv2.putText(frame, 'Rack 1 Third Row Product_4 - Availble ', (50, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[14] == "False":
        cv2.putText(frame, 'Rack 1 Third Row Product_4 - Not Availble ', (50, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[15] == "True":
        cv2.putText(frame, 'Rack 1 Third Row Product_5 - Availble ', (50, 750),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[15] == "False":
        cv2.putText(frame, 'Rack 1 Third Row Product_5 - Not Availble ', (50, 750),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[16] == "True":
        cv2.putText(frame, 'Rack 2 First Row Product_1 - Availble ', (1200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[16] == "False":
        cv2.putText(frame, 'Rack 2 First Row Product_1 - Not Availble ', (1200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[17] == "True":
        cv2.putText(frame, 'Rack 2 First Row Product_2 - Availble ', (1200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[17] == "False":
        cv2.putText(frame, 'Rack 2 First Row Product_2 - Not Availble ', (1200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[18] == "True":
        cv2.putText(frame, 'Rack 2 First Row Product_3 - Availble ', (1200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[18] == "False":
        cv2.putText(frame, 'Rack 2 First Row Product_3 - Not Availble ', (1200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[19] == "True":
        cv2.putText(frame, 'Rack 2 Second Row Product_1 - Availble ', (1200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[19] == "False":
        cv2.putText(frame, 'Rack 2 Second Row Product_1 - Not Availble ', (1200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[20] == "True":
        cv2.putText(frame, 'Rack 2 Second Row Product_2 - Availble ', (1200, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[20] == "False":
        cv2.putText(frame, 'Rack 2 Second Row Product_2 - Not Availble ', (1200, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[21] == "True":
        cv2.putText(frame, 'Rack 2 Second Row Product_3 - Availble ', (1200, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[21] == "False":
        cv2.putText(frame, 'Rack 2 Second Row Product_3 - Not Availble ', (1200, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[22] == "True":
        cv2.putText(frame, 'Rack 2 Third Row Product_1  - Availble ', (1200, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[22] == "False":
        cv2.putText(frame, 'Rack 2 Third Row Product_1 - Not Availble ', (1200, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[23] == "True":
        cv2.putText(frame, 'Rack 2 Third Row Product_2 - Availble ', (1200, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[23] == "False":
        cv2.putText(frame, 'Rack 2 Third Row Product_2 - Not Availble ', (1200, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if dict_s[24] == "True":
        cv2.putText(frame, 'Rack 2 Third Row Product_3 - Availble ', (1200, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    elif dict_s[24] == "False":
        cv2.putText(frame, 'Rack 2 Third Row Product_3 - Not Availble ', (1200, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    
    count_list_rack1 = list(dict_s.values())[:15]
    counts1 = count_list_rack1.count("True")
    # print(counts,":::")
    cv2.putText(frame, "Rack 1 Count - " + str(counts1), (50, 1000),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)
    
    count_list_rack2 = list(dict_s.values())[15:]
    counts2 = count_list_rack2.count("True")
    # print(counts,":::")
    cv2.putText(frame, "Rack 2 Count - " + str(counts2), (1200, 1000),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)
    
    cv2.rectangle(frame, (618, 1049), (991, 1247), (255, 0, 0), 2)
    cv2.putText(frame, "Rack - 1" , (1000, 1122),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)

    cv2.rectangle(frame, (736, 1240),(1024, 1431), (255, 0, 0), 2)
    cv2.putText(frame, "Rack-2", (1018, 1300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)
    
    end = time.time()
    diff = end - start
    if diff > 0:
        cv2.putText(frame, f"FPS ~ {1 / diff: .3f}", (25, 1400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
    result.write(frame)
    cv2.imshow("frame", cv2.resize(frame, None, fx=0.5, fy=0.5))
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

result.release()
cv2.destroyAllWindows()
