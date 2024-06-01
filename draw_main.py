import cv2
import numpy as np
import os
import webbrowser
from tkinter import filedialog, Tk, Text, Button
from datetime import datetime
from motion_recognition import finger_movement

class SmartBoard:
    def __init__(self):
        # Your existing initialization code here
        self.text_editor_open = False
        self.sw, self.sh = 800, 600
        self.canvas = np.zeros((self.sh, self.sw, 3), np.uint8)
        self.color = (0, 0, 255)
        self.thickness = 5
        self.eraser_thickness = 100
        self.x_prev, self.y_prev = 0, 0
        self.recording = False
        self.out = None
        self.header_images = self.load_header_images()
        self.hh, self.hw, _ = self.header_images[0].shape
        self.header = 0

        self.cam = cv2.VideoCapture(0)
        self.detector = finger_movement()

        self.text_editor_window = None
        self.text_editor = None

    def load_header_images(self):
        header_images = []
        image_folder = "C:\\Users\\saite\\OneDrive\\Desktop\\project\\newvishnuproject\\Project\\images"
        for i in os.listdir(image_folder):
            img_path = os.path.join(image_folder, i)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.sw, int(self.sw * (img.shape[0] / img.shape[1]))))
                header_images.append(img)
            else:
                print(f"Failed to load image: {img_path}")
        return header_images

    def open_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ('.png', '.jpg', '.jpeg'):
            # Handle image files using OpenCV
            img = cv2.imread(file_path)
            resized_img = cv2.resize(img, (600, 400))

            if img is not None:
                cv2.imshow("Output", resized_img)
            else:
                print(f"Failed to read the image from {file_path}")
        else:
            # Open other file types using the default application
            webbrowser.open(file_path)

    def open_text_editor(self):
        self.text_editor_window = Tk()
        self.text_editor_window.title("Text Editor")
        self.text_editor = Text(self.text_editor_window)
        self.text_editor.pack()

        save_button = Button(self.text_editor_window, text="Save", command=self.save_text)
        save_button.pack()
        print("Text editor opened.")
        self.text_editor_open = True
        self.text_editor_window.mainloop()

    def save_text(self):
        text_content = self.text_editor.get("1.0", "end-1c")  # Get text content from Text widget
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")  # Ask user for file path to save
        if file_path:
            with open(file_path, "w") as file:
                file.write(text_content)
            print("Text saved successfully.")

    def draw_square(self, frame):
        # Function to draw a square outline using hand gestures
        try:
            if self.detector.listD:
                index_finger_position = self.detector.finger_tip(frame, draw=False)[8][1:]
                thumb_position = self.detector.finger_tip(frame, draw=False)[4][1:]
                size = abs(index_finger_position[0] - thumb_position[0])
                x1 = index_finger_position[0] - size // 2
                y1 = index_finger_position[1] - size // 2
                x2 = index_finger_position[0] + size // 2
                y2 = index_finger_position[1] + size // 2
                cv2.rectangle(self.canvas, (x1, y1), (x2, y2), self.color, self.thickness)
        except:        
            #else:
            print("Finger tip detection failed. Unable to draw square.")

    def run(self):
        while True:
            status, frame = self.cam.read()
            frame = cv2.resize(frame, (self.sw, self.sh))
            frame = cv2.flip(frame, 1)
            frame = self.detector.hand_recognition(frame, draw=False)
            listD = self.detector.finger_tip(frame, draw=False)

            if listD:
                x1, y1 = listD[8][1:]
                x2, y2 = listD[12][1:]

                fingers = self.detector.all_fingers()

                if fingers[1] and not fingers[2]:
                    cv2.circle(frame, (x1, y1), 8, self.color, -1)
                    if self.x_prev == 0 and self.y_prev == 0:
                        self.x_prev, self.y_prev = x1, y1

                    if self.color == (0, 0, 0):
                        et_h = self.eraser_thickness // 2
                        cv2.rectangle(frame, (x1 - et_h, y1 - et_h), (x1 + et_h, y1 + et_h), (0, 0, 0), -1)
                        cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), self.color, self.eraser_thickness)
                    else:
                        cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), self.color, self.thickness)

                    self.x_prev, self.y_prev = x1, y1

                if fingers[1] and fingers[2]:
                    self.x_prev, self.y_prev = 0, 0
                    if y1 < self.hh:
                        self.update_color_based_on_x_position(x1)

                finger_positions = self.detector.all_fingers()
                if all(finger_positions):
                    et_h = self.eraser_thickness // 2
                    cv2.rectangle(frame, (x1 - et_h, y1 - et_h), (x1 + et_h, y1 + et_h), (0, 0, 0), -1)
                    cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), (0, 0, 0), self.eraser_thickness*2)

            if not self.recording:
                self.out = None
            elif self.recording and not self.out:
                self.start_recording()

            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

            img_inv = cv2.resize(img_inv, (self.sw, self.sh))

            frame = cv2.bitwise_and(frame, img_inv)

            self.canvas = cv2.resize(self.canvas, (self.sw, self.sh))
            frame = cv2.bitwise_or(frame, self.canvas)

            self.draw_header(frame)

            if self.recording and self.out:
                self.record_frame(frame)

            cv2.imshow("SmartBoard", frame)

            key = cv2.waitKey(1)

            if key == ord('o'):
                filepath = filedialog.askopenfilename(initialdir="C:\\Users\\hp\\Downloads", title="Open",
                                                       filetypes=(("text files", "*.txt"), ("all files", "*.*")))
                self.open_file(filepath)

            if key == ord('c'):
                self.canvas = np.zeros((self.sh, self.sw, 3), np.uint8)
                self.x_prev = 0
                self.y_prev = 0

            if key == ord('s'):
                self.draw_square(frame)

            if key == ord('t'):  # Open text editor
                #self.open_text_editor()
                if not self.text_editor_open:  # Check if text editor is not already open
                    self.open_text_editor()

            cv2.imshow("SmartBoard", frame)

            if key == ord('q'):
                self.release_cam()
                break

        cv2.destroyAllWindows()

    def update_color_based_on_x_position(self, x):
        if 120 <= x <= 240:
            self.color = (0, 0, 255)
            self.header = 0
        elif 260 <= x <= 380:
            self.color = (255, 0, 0)
            self.header = 1
        elif 410 <= x <= 530:
            self.color = (0, 255, 0)
            self.header = 2
        elif 540 <= x <= 660:
            self.color = (156, 0, 210)
            self.header = 3
        elif x >= 670:
            self.color = (0, 0, 0)
            self.header = 4
        elif x <= 100:
            self.recording = True

    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        video_name = f"C:/Users/rachu/OneDrive/Desktop/newvishnuproject/Project/recordings/{now}.mp4"
        self.out = cv2.VideoWriter(video_name, fourcc, 30, (self.sw, self.sh))

    def draw_header(self, frame):
        frame[:self.hh, :self.hw] = self.header_images[self.header]

    def record_frame(self, frame):
        cv2.rectangle(frame, (0, 0), (100, self.hh), (41, 156, 0), 3)
        self.out.write(frame)

    def release_cam(self):
        self.cam.release()
        if self.out:
            self.out.release()


if __name__ == "__main__":
    smart_board = SmartBoard()
    smart_board.run()



"""import cv2
import numpy as np
import os
import webbrowser
from tkinter import filedialog, Tk, Text, Button
from datetime import datetime
from motion_recognition import finger_movement

class SmartBoard:
    def __init__(self):
        # Your existing initialization code here
        self.text_editor_open = False
        self.sw, self.sh = 800, 600
        self.canvas = np.zeros((self.sh, self.sw, 3), np.uint8)
        self.color = (0, 0, 255)
        self.thickness = 5
        self.eraser_thickness = 100
        self.x_prev, self.y_prev = 0, 0
        self.recording = False
        self.out = None
        self.header_images = self.load_header_images()
        self.hh, self.hw, _ = self.header_images[0].shape
        self.header = 0

        self.cam = cv2.VideoCapture(0)
        self.detector = finger_movement()

        self.text_editor_window = None
        self.text_editor = None
        self.whiteboard_canvas = np.zeros((self.sh, self.sw, 3), np.uint8)  # Initialize the whiteboard canvas


    def load_header_images(self):
        header_images = []
        image_folder = 'C:/Users/rachu/OneDrive/Desktop/vishnuproject/Project/images'
        for i in os.listdir(image_folder):
            img_path = os.path.join(image_folder, i)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.sw, int(self.sw * (img.shape[0] / img.shape[1]))))
                header_images.append(img)
            else:
                print(f"Failed to load image: {img_path}")
        return header_images

    def open_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ('.png', '.jpg', '.jpeg'):
            # Handle image files using OpenCV
            img = cv2.imread(file_path)
            resized_img = cv2.resize(img, (600, 400))

            if img is not None:
                cv2.imshow("Output", resized_img)
            else:
                print(f"Failed to read the image from {file_path}")
        else:
            # Open other file types using the default application
            webbrowser.open(file_path)

    def open_text_editor(self):
        self.text_editor_window = Tk()
        self.text_editor_window.title("Text Editor")
        self.text_editor = Text(self.text_editor_window)
        self.text_editor.pack()

        save_button = Button(self.text_editor_window, text="Save", command=self.save_text)
        save_button.pack()
        print("Text editor opened.")
        self.text_editor_open = True
        self.text_editor_window.mainloop()

    def save_text(self):
        text_content = self.text_editor.get("1.0", "end-1c")  # Get text content from Text widget
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")  # Ask user for file path to save
        if file_path:
            with open(file_path, "w") as file:
                file.write(text_content)
            print("Text saved successfully.")

    def draw_square(self, frame):
        # Function to draw a square outline using hand gestures
        try:
            if self.detector.listD:
                index_finger_position = self.detector.finger_tip(frame, draw=False)[8][1:]
                thumb_position = self.detector.finger_tip(frame, draw=False)[4][1:]
                size = abs(index_finger_position[0] - thumb_position[0])
                x1 = index_finger_position[0] - size // 2
                y1 = index_finger_position[1] - size // 2
                x2 = index_finger_position[0] + size // 2
                y2 = index_finger_position[1] + size // 2
                cv2.rectangle(self.canvas, (x1, y1), (x2, y2), self.color, self.thickness)
        except:        
            #else:
            print("Finger tip detection failed. Unable to draw square.")

    def run(self):
        while True:
            status, frame = self.cam.read()
            frame = cv2.resize(frame, (self.sw, self.sh))
            frame = cv2.flip(frame, 1)
            frame = self.detector.hand_recognition(frame, draw=False)
            listD = self.detector.finger_tip(frame, draw=False)
            #self.draw_square(frame)
            shortened_frame = frame[:, 150:self.sw - 150]
            # Display the whiteboard canvas along with the shortened frame
            whiteboard_display = np.hstack((shortened_frame, self.whiteboard_canvas))
            cv2.imshow("SmartBoard", whiteboard_display)
            if listD:
                x1, y1 = listD[8][1:]
                x2, y2 = listD[12][1:]

                fingers = self.detector.all_fingers()

                if fingers[1] and not fingers[2]:
                    cv2.circle(frame, (x1, y1), 8, self.color, -1)
                    if self.x_prev == 0 and self.y_prev == 0:
                        self.x_prev, self.y_prev = x1, y1

                    if self.color == (0, 0, 0):
                        et_h = self.eraser_thickness // 2
                        cv2.rectangle(frame, (x1 - et_h, y1 - et_h), (x1 + et_h, y1 + et_h), (0, 0, 0), -1)
                        cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), self.color, self.eraser_thickness)
                    else:
                        cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), self.color, self.thickness)

                    self.x_prev, self.y_prev = x1, y1

                if fingers[1] and fingers[2]:
                    self.x_prev, self.y_prev = 0, 0
                    if y1 < self.hh:
                        self.update_color_based_on_x_position(x1)

                finger_positions = self.detector.all_fingers()
                if all(finger_positions):
                    et_h = self.eraser_thickness // 2
                    cv2.rectangle(frame, (x1 - et_h, y1 - et_h), (x1 + et_h, y1 + et_h), (0, 0, 0), -1)
                    cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), (0, 0, 0), self.eraser_thickness*2)

            if not self.recording:
                self.out = None
            elif self.recording and not self.out:
                self.start_recording()

            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

            img_inv = cv2.resize(img_inv, (self.sw, self.sh))

            frame = cv2.bitwise_and(frame, img_inv)

            self.canvas = cv2.resize(self.canvas, (self.sw, self.sh))
            frame = cv2.bitwise_or(frame, self.canvas)

            self.draw_header(frame)

            if self.recording and self.out:
                self.record_frame(frame)

            cv2.imshow("SmartBoard", frame)

            key = cv2.waitKey(1)

            if key == ord('o'):
                filepath = filedialog.askopenfilename(initialdir="C:\\Users\\hp\\Downloads", title="Open",
                                                       filetypes=(("text files", "*.txt"), ("all files", "*.*")))
                self.open_file(filepath)

            if key == ord('c'):
                self.canvas = np.zeros((self.sh, self.sw, 3), np.uint8)
                self.x_prev = 0
                self.y_prev = 0

            if key == ord('s'):
                self.draw_square(frame)

            if key == ord('t'):  # Open text editor
                #self.open_text_editor()
                if not self.text_editor_open:  # Check if text editor is not already open
                    self.open_text_editor()

            cv2.imshow("SmartBoard", frame)
            
            if key == ord('q'):
                self.release_cam()
                break

        cv2.destroyAllWindows()

    def update_color_based_on_x_position(self, x):
        if 120 <= x <= 240:
            self.color = (0, 0, 255)
            self.header = 0
        elif 260 <= x <= 380:
            self.color = (255, 0, 0)
            self.header = 1
        elif 410 <= x <= 530:
            self.color = (0, 255, 0)
            self.header = 2
        elif 540 <= x <= 660:
            self.color = (156, 0, 210)
            self.header = 3
        elif x >= 670:
            self.color = (0, 0, 0)
            self.header = 4
        elif x <= 100:
            self.recording = True

    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        video_name = f"C:/Users/rachu/OneDrive/Desktop/newvishnuproject/Project/recordings/{now}.mp4"
        self.out = cv2.VideoWriter(video_name, fourcc, 30, (self.sw, self.sh))

    def draw_header(self, frame):
        frame[:self.hh, :self.hw] = self.header_images[self.header]

    def record_frame(self, frame):
        cv2.rectangle(frame, (0, 0), (100, self.hh), (41, 156, 0), 3)
        self.out.write(frame)

    def release_cam(self):
        self.cam.release()
        if self.out:
            self.out.release()


if __name__ == "__main__":
    smart_board = SmartBoard()
    smart_board.run()"""






"""import cv2
import numpy as np
import os
import webbrowser
from tkinter import filedialog
from datetime import datetime
from motion_recognition import finger_movement

class SmartBoard:
    def __init__(self):
        self.sw, self.sh = 800, 600
        self.canvas = np.zeros((self.sh, self.sw, 3), np.uint8)
        self.color = (0, 0, 255)
        self.thickness = 5
        self.eraser_thickness = 100
        self.x_prev, self.y_prev = 0, 0
        self.recording = False
        self.out = None
        self.header_images = self.load_header_images()
        self.hh, self.hw, _ = self.header_images[0].shape
        self.header = 0

        self.cam = cv2.VideoCapture(0)
        self.detector = finger_movement()

        # New attributes for image opening
        # self.image_opening = False
        # self.selected_image = None

    def load_header_images(self):
        header_images = []
        image_folder = 'C:/Users/rachu/OneDrive/Desktop/vishnuproject/Project/images'
        for i in os.listdir(image_folder):
            img_path = os.path.join(image_folder, i)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.sw, int(self.sw * (img.shape[0] / img.shape[1]))))
                header_images.append(img)
            else:
                print(f"Failed to load image: {img_path}")
        return header_images

    # def open_image(self, image_path):
    #     c = cv2.imread(image_path)
    #     cv2.imshow("blah", c)
    #     self.image_opening = False
    #     self.selected_image = None

    def open_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ('.png', '.jpg', '.jpeg'):
            # Handle image files using OpenCV
            img = cv2.imread(file_path)
            resized_img = cv2.resize(img, (600, 400))

            if img is not None:
                cv2.imshow("Output", resized_img)
            else:
                print(f"Failed to read the image from {file_path}")
        else:
            # Open other file types using the default application
            webbrowser.open(file_path)

    def draw_square(self, frame):
    # Function to draw a square outline using hand gestures
        # fingers_2 = self.detector.all_fingers()

    # Ensure that finger tip detection is successful
        if self.detector.listD:
        # Get the position of the index finger
            index_finger_position = self.detector.finger_tip(frame, draw=False)[8][1:]

        # Get the position of the thumb
            thumb_position = self.detector.finger_tip(frame, draw=False)[4][1:]

        # Calculate the size of the square based on the distance between index finger and thumb
            size = abs(index_finger_position[0] - thumb_position[0])

        # Calculate the top-left corner coordinates of the square
            x1 = index_finger_position[0] - size // 2
            y1 = index_finger_position[1] - size // 2

        # Calculate the bottom-right corner coordinates of the square
            x2 = index_finger_position[0] + size // 2
            y2 = index_finger_position[1] + size // 2

        # Draw the square outline on the canvas
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), self.color, self.thickness)
        else:
            print("Finger tip detection failed. Unable to draw square.")


    def run(self):
        while True:
            status, frame = self.cam.read()
            frame = cv2.resize(frame, (self.sw, self.sh))
            frame = cv2.flip(frame, 1)
            frame = self.detector.hand_recognition(frame, draw=False)
            listD = self.detector.finger_tip(frame, draw=False)

            if listD:
                x1, y1 = listD[8][1:]
                x2, y2 = listD[12][1:]

                fingers = self.detector.all_fingers()

                if fingers[1] and not fingers[2]:
                    cv2.circle(frame, (x1, y1), 8, self.color, -1)
                    if self.x_prev == 0 and self.y_prev == 0:
                        self.x_prev, self.y_prev = x1, y1

                    if self.color == (0, 0, 0):
                        et_h = self.eraser_thickness // 2
                        cv2.rectangle(frame, (x1 - et_h, y1 - et_h), (x1 + et_h, y1 + et_h), (0, 0, 0), -1)
                        cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), self.color, self.eraser_thickness)
                    else:
                        cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), self.color, self.thickness)

                    self.x_prev, self.y_prev = x1, y1

                if fingers[1] and fingers[2]:
                    self.x_prev, self.y_prev = 0, 0
                    if y1 < self.hh:
                        self.update_color_based_on_x_position(x1)

                finger_positions = self.detector.all_fingers()
                if all(finger_positions):
                    et_h = self.eraser_thickness // 2
                    cv2.rectangle(frame, (x1 - et_h, y1 - et_h), (x1 + et_h, y1 + et_h), (0, 0, 0), -1)
                    cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), (0, 0, 0), self.eraser_thickness*2)

            if not self.recording:
                self.out = None
            elif self.recording and not self.out:
                self.start_recording()

            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

            # Ensure img_inv has the same dimensions as frame
            img_inv = cv2.resize(img_inv, (self.sw, self.sh))

            frame = cv2.bitwise_and(frame, img_inv)

            # Ensure self.canvas has the same dimensions as frame
            self.canvas = cv2.resize(self.canvas, (self.sw, self.sh))
            frame = cv2.bitwise_or(frame, self.canvas)

            self.draw_header(frame)

            if self.recording and self.out:
                self.record_frame(frame)

            cv2.imshow("SmartBoard", frame)

            # Check for 'i' key to open image
            key = cv2.waitKey(1)
            # if key == ord('i'):
            #     self.image_opening = True
            #     self.selected_image = r"C:\\Users\\hp\\OneDrive\\Desktop\\Internship\\Personal\\Project\\images\\body.jpg"  # Set the path to the image you want to open

            if key == ord('o'):
                filepath = filedialog.askopenfilename(initialdir="C:\\Users\\hp\\Downloads",title="Open",filetypes=(("text files","*.txt"), ("all files","*.*")))
                self.open_file(filepath)
                # c = cv2.imread(filepath)
                # cS = cv2.resize(c, (600, 400))
                # cv2.imshow("Output", c)

            # if self.selected_image:
            #     self.open_image(self.selected_image)
            #     self.selected_image = None

            # if self.image_opening:
            #     self.draw_centered_image()

            if key == ord('c'):
                self.canvas= np.zeros((self.sh, self.sw, 3), np.uint8)
                self.x_prev= 0
                self.y_prev= 0

            if key == ord('s'):  # Check for 's' key to draw square
                self.draw_square(frame)

            if key == ord('q'):  # Check for 'q' key to exit
                self.release_cam()
                break

        cv2.destroyAllWindows()

    # def draw_centered_image(self):
    #     if self.canvas is not None:
    #         h, w, _ = self.canvas.shape
    #         x_offset = (self.sw - w) // 2
    #         y_offset = (self.sh - h) // 2

    #         # Ensure the offsets are non-negative
    #         x_offset = max(0, x_offset)
    #         y_offset = max(0, y_offset)

    #         # Calculate the region to display the image
    #         region = self.canvas[y_offset:y_offset + h, x_offset:x_offset + w]

    #         # Copy the region to the frame
    #         frame[y_offset:y_offset + h, x_offset:x_offset + w] = region

    def update_color_based_on_x_position(self, x):
        if 120 <= x <= 240:
            self.color = (0, 0, 255)
            self.header = 0
        elif 260 <= x <= 380:
            self.color = (255, 0, 0)
            self.header = 1
        elif 410 <= x <= 530:
            self.color = (0, 255, 0)
            self.header = 2
        elif 540 <= x <= 660:
            self.color = (156, 0, 210)
            self.header = 3
        elif x >= 670:
            self.color = (0, 0, 0)
            self.header = 4
        elif x<= 100:
            self.recording= True

    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #video_name = f"recordings/{now}.mp4"
        video_name = f"C:/Users/rachu/OneDrive/Desktop/newvishnuproject/Project/recordings/{now}.mp4"
        self.out = cv2.VideoWriter(video_name, fourcc, 30, (self.sw, self.sh))

    def draw_header(self, frame):
        frame[:self.hh, :self.hw] = self.header_images[self.header]

    def record_frame(self, frame):
        cv2.rectangle(frame, (0, 0), (100, self.hh), (41, 156, 0), 3)
        self.out.write(frame)

    def release_cam(self):
        self.cam.release()
        if self.out:
            self.out.release()

if __name__ == "__main__":
    smart_board = SmartBoard()
    smart_board.run()"""
