import cv2
from tracker import ObjectCounter  # Importing ObjectCounter from tracker.py

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check for mouse movement
        point = [x, y]
        print(f"Mouse moved to: {point}")

# Open the video file
cap = cv2.VideoCapture('tf.mp4')

# Define region points for counting
region_points = [(0, 112), (224, 112)]
# Initialize the object counter
counter = ObjectCounter(
    region=region_points,  # Pass region points
    model="yolo11n_ncnn_model",  # Model for object counting
    classes=[2,5,7],  # Detect only cars,truck,motorcycles
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust line width for display
)

# Create a named window and set the mouse callback


count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
        # If video ends, reset to the beginning
#        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#        continue
    count += 1
    if count % 5 != 0:  # Skip odd frames
        continue

    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_NEAREST)

    # Process the frame with the object counter
    frame1 = counter.count(frame)
   
    # Show the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
