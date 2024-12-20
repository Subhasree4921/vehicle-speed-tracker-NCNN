import cv2
from tracker import ObjectCounter  # Importing ObjectCounter from tracker.py

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(f"Mouse moved to: {point}")


cap = cv2.VideoCapture(0)

# Define region points for counting
region_points = [(0, 112), (224, 112)]

counter = ObjectCounter(
    region=region_points,  
    model="yolo11n_ncnn_model",  
    classes=[2,5,7], 
    show_in=True,  
    show_out=True,  
    line_width=2,  
)


count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
        
    count += 1
    if count % 5 != 0:  
        continue

    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_NEAREST)

    # Process the frame with the object counter
    frame1 = counter.count(frame)
   
    # Show the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break


cap.release()
cv2.destroyAllWindows()
