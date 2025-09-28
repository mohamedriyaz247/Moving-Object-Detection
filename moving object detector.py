import cv2
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Threshold the foreground mask
    _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Moving Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release webcam capture and close windows
cap.release()
cv2.destroyAllWindows()
