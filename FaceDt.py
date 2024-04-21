import cv2

cap = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in detections:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165,255), 2)
        text = f'Face: ({x}, {y})'
        text = f'Name: (Mayank)'                            #Showcasing Name Above Your Image 
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):           ## To exit the Frame or Screen Press Q 
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
