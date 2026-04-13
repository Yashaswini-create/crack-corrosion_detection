import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import glob

# Ask user input
while True:
    mode = input("Enter 'c' for corrosion, 'k' for crack (NEU-DET), or 's' for small custom crack dataset (no annotations): ").strip().lower()
    if mode in ['c', 'k', 's']:
        break
    print("Invalid input. Please enter 'c', 'k', or 's'.")

# Set image paths
if mode == 'c':
    image_folder = r"D:\drive\datasets\corrosion detect\corrosion detect\images"
    label_folder = r"D:\drive\datasets\corrosion detect\corrosion detect\labels"
    label_name = 'corrosion'
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(valid_exts)]

elif mode == 'k':
    image_folder = r"D:\drive\datasets\crack detect\NEU-DET\train\images"
    label_folder = r"D:\drive\datasets\crack detect\NEU-DET\train\annotations"
    label_name = 'crack'
    valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_files = []
    for ext in valid_exts:
        image_files.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))

elif mode == 's':
    image_folder = r"D:\drive\datasets\crack_small_dataset"
    label_name = 'crack'
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(valid_exts)]

if not image_files:
    print(f"No images found in: {image_folder}")
    exit()

# Resize helper
def resize_to_fit(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

# Draw Ground Truth Box (used only for mode 'k')
def draw_ground_truth_box(image, xml_path, label_name='crack'):
    if not os.path.exists(xml_path):
        return image
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if label_name not in name:
                continue
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(image, 'GT', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    except:
        pass
    return image

# Main loop
for image_path in image_files:
    image_file = os.path.basename(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load: {image_file}")
        continue

    print(f"Processing: {image_file}")
    image_resized = cv2.resize(image, (640, 480))

    if mode == 'k':
        # ---- CRACK DETECTION (with annotation) ----
        gt_image = image_resized.copy()
        xml_file = os.path.splitext(image_file)[0] + ".xml"
        xml_path = os.path.join(label_folder, xml_file)
        gt_image = draw_ground_truth_box(gt_image, xml_path, label_name)

        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        mask_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        detection_image = image_resized.copy()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(detection_image, contours, -1, (0, 255, 0), 1)

        combined = np.hstack((gt_image, detection_image, mask_bgr))
        cv2.namedWindow("Crack (GT + Detection + Edge)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Crack (GT + Detection + Edge)", 1280, 600)
        cv2.imshow("Crack (GT + Detection + Edge)", resize_to_fit(combined, 1600, 600))

    elif mode == 's':
        # ---- SMALL CUSTOM CRACK DATASET (no GT) ----
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        mask_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        detection_image = image_resized.copy()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(detection_image, contours, -1, (0, 255, 0), 1)

        combined = np.hstack((image_resized, detection_image, mask_bgr))
        cv2.namedWindow("Custom Crack Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Custom Crack Detection", 1280, 600)
        cv2.imshow("Custom Crack Detection", resize_to_fit(combined, 1600, 600))

    elif mode == 'c':
        # ---- CORROSION DETECTION ----
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        lower_rust = np.array([0, 50, 50])
        upper_rust = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_rust, upper_rust)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        detection_image = image_resized.copy()
        detection_image[mask > 0] = (0, 0, 255)

        combined = np.hstack((image_resized, detection_image))
        cv2.namedWindow("Corrosion: Original | Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Corrosion: Original | Detection", 1280, 480)
        cv2.imshow("Corrosion: Original | Detection", resize_to_fit(combined, 1280, 480))

        cv2.namedWindow("Corrosion Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Corrosion Mask", 640, 480)
        cv2.imshow("Corrosion Mask", mask)

    print(f"[{image_file}] - Press any key for next image or 'q' to quit...")
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
