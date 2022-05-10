import cv2

img = cv2.imread('zam/sample.jpg')

with open('zam/obj.names', 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('zam/yolov4-obj.cfg', 'zam/yolov4-obj_last.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=2)
    classes_1 = classes[classId]
    text = classes_1 + str(score)
    # text = '%s: %.2f' % (classes[classId[0]], score)
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 255, 0), thickness=2)

cv2.imshow('Image', img)
filename = 'output.jpg'
cv2.imwrite(filename, img)
cv2.waitKey(0)
cv2.destroyAllWindows()