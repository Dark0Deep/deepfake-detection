import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_faces(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for img in os.listdir(input_folder):

        img_path = os.path.join(input_folder, img)
        image = cv2.imread(img_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for i,(x,y,w,h) in enumerate(faces):
            face = image[y:y+h, x:x+w]

            filename = os.path.join(
                output_folder,
                f"{img}_{i}.jpg"
            )

            cv2.imwrite(filename, face)

print("Extracting Train Real Faces...")
extract_faces(
    "../dataset_split/train/real",
    "../faces/train/real"
)

print("Extracting Train Fake Faces...")
extract_faces(
    "../dataset_split/train/fake",
    "../faces/train/fake"
)

print("Extracting Validation Real Faces...")
extract_faces(
    "../dataset_split/validation/real",
    "../faces/validation/real"
)

print("Extracting Validation Fake Faces...")
extract_faces(
    "../dataset_split/validation/fake",
    "../faces/validation/fake"
)
print("Extracting Test Real Faces...")
extract_faces(
    "../dataset_split/test/real",
    "../faces/test/real"
)

print("Extracting Test Fake Faces...")
extract_faces(
    "../dataset_split/test/fake",
    "../faces/test/fake"
)   

print("Done")