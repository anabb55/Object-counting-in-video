import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import mean_absolute_error
import sys


positive_imgs = []
negative_imgs = []
positive_features = []
negative_features = []
labels = []
bins = 9
cell_size = (8,8)
block_size=(3,3)
true_values = []
my_values = []

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        
        sys.exit(1)

    data_folder = sys.argv[1]

    images_dir = os.path.join(data_folder, 'pictures')
    videos_folder = os.path.join(data_folder, 'videos')
    csv_path = os.path.join(data_folder, 'counts.csv')

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

for name_img in os.listdir(images_dir):
    path = os.path.join(images_dir, name_img)
    image = load_image(path)
    if 'p_' in name_img:
        positive_imgs.append(image)
    elif 'n_' in name_img:
        negative_imgs.append(image)


## zapamti da winSize mora biti deljiv sa velicinom celija u pikselima, pa zbog toga imamo ovo racunanje
## takodje nam treba velicina bloka u pikselima
hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1], 
                                  image.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=bins)


for image in positive_imgs:
    positive_features.append(hog.compute(image)) 
    labels.append(1)

for image in negative_imgs:
    negative_features.append(hog.compute(image))
    labels.append(0)

positive_features = np.array(positive_features) 
negative_features = np.array(negative_features) 

x = np.vstack((positive_features, negative_features))
y = np.array(labels)



## obucavanje SVM klasifikatora 
clf_svm = SVC(kernel='linear', probability=True)  ## pored toga sto govori da li je nesto auto ili ne, on govori kolika je verovatnoca pripadnosti toj klasi
clf_svm.fit(x, y)

def classify_window(window):
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]


## detektovanje linije
def line_detector(image):

 hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    
 lower_red = np.array([0, 100, 100])
 upper_red = np.array([70, 255, 255])

   
 red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
 red_pixels = cv2.bitwise_and(image, image, mask=red_mask)

 img_gray = cv2.cvtColor(red_pixels, cv2.COLOR_BGR2GRAY)
 img_canny = cv2.Canny(img_gray, 50, 150, apertureSize=3)

 min_length = 100

 lines = cv2.HoughLinesP(image=img_canny, rho=1, theta=np.pi/180, threshold=15, lines=np.array([]), minLineLength=min_length, maxLineGap=200)

 if lines is not None and len(lines) > 0:
    x1 = lines[0][0][0]
    y1 = lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = lines[0][0][3]
    
    # print(x1, y1, x2, y2)
 else:
    print("Linija nije detektovana.")

 return (x1, y1, x2, y2)



##izdvajanje auta 
def process_image(image, step_size_y, step_size_x, window_size=(500,200), threshold=0.9):
    detected_windows = []
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for y in range(0, rgb_image.shape[0], step_size_y):
        for x in range(0, rgb_image.shape[1], step_size_x):
            this_window=(y,x)
            window = rgb_image[y:y+window_size[1], x:x+window_size[0]]
            
            if(window.shape == (window_size[1], window_size[0],3)):
                resized_window = cv2.resize(window, (120, 60))
               
                score = classify_window(resized_window)

                if score>threshold:
                   detected_windows.append((this_window, score))

    return detected_windows
   



def process_video(path):
    frame_num = 0
    counter = 0
    cap = cv2.VideoCapture(path)

    cap.set(1, frame_num)  

    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        if not grabbed:
            break  

        line_detectorr = line_detector(frame)

        line_up_y = line_detectorr[3]
        line_down_y = line_detectorr[1]
        

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(rgb_frame)
        # plt.show()
        detected_windows = process_image(rgb_frame, 60,130)
        grouped_windows = {}
        for window, score in detected_windows:
            y_coord = window[0]

            if y_coord not in grouped_windows:
                grouped_windows[y_coord] = []
            grouped_windows[y_coord].append((window, score))

        best_windows = []
        for y_coord, windows_in_group in grouped_windows.items():
            best_window_in_group = max(windows_in_group, key=lambda x: x[1])
            best_windows.append(best_window_in_group)

      

        for win, score in best_windows:
            center_x = win[1] + 235
            center_y = win[0] + 100
            # print(abs(center_x - line_detectorr[0]))


            if (line_up_y < center_y < line_down_y) and ((abs(center_x - line_detectorr[0])) < 180):
                # print("Detektovanoooo")
                counter += 1

    # Return counter nakon što se završi obrada svih frejmova
    return counter


with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  
    for row in csv_reader:
        video_name, count = row
        true_values.append(int(count))

    

for i, filename in enumerate(os.listdir(videos_folder)):
    if filename.endswith(".mp4"):  
        video_path = os.path.join(videos_folder, filename)
        
        
        br = process_video(video_path)
        my_values.append(br)
        print(f"{filename} - {true_values[i]} - {br}")

#MAE
mae = mean_absolute_error(true_values, my_values)

print("Ukupan MAE je: {:.2f}".format(mae))







