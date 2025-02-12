import os
import cv2
import glob
import numpy as np
import albumentations as A
from multiprocessing import Pool, cpu_count

# Percorsi del dataset
img_dir = "/Users/mariocorrente/Desktop/plantDetector-master/datasets/WeedCrop.v1i.yolov5pytorch/train/images"
label_dir = "/Users/mariocorrente/Desktop/plantDetector-master/datasets/WeedCrop.v1i.yolov5pytorch/train/labels"
output_img_dir = "/Users/mariocorrente/Desktop/df_aum/newDataset/images"
output_label_dir = "/Users/mariocorrente/Desktop/df_aum/newDataset/labels"

# Crea cartelle output se non esistono
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Conta il numero di immagini originali
image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
num_original_images = len(image_paths)
target_images = 15000  # Aumentiamo a 15k immagini

# Determina il fattore di moltiplicazione per raggiungere il target
augmentation_factor = max(target_images // num_original_images, 1)

print(f"Immagini originali: {num_original_images}")
print(f"Ogni immagine sarà aumentata ~{augmentation_factor} volte")

# Definizione delle trasformazioni di Data Augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),  
    A.GaussianBlur(blur_limit=3, p=0.2),  
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  
    A.HueSaturationValue(p=0.3),  
    A.Affine(shear=(-5, 5), p=0.3),  # Shear ridotto per meno distorsione
    A.ElasticTransform(alpha=0.5, sigma=30, alpha_affine=30, p=0.1),  # Alleggerito
    A.CoarseDropout(max_holes=2, max_height=30, max_width=30, p=0.2),  
    A.Resize(640, 640),  
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Funzione per processare una singola immagine
def process_image(img_path):
    filename = os.path.basename(img_path)
    label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        return  

    # Carica immagine e label
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes, class_labels = [], []
    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        x_center, y_center, w, h = map(float, values[1:])
        bboxes.append([x_center, y_center, w, h])
        class_labels.append(class_id)

    # Genera immagini aumentate
    for i in range(augmentation_factor):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        # Salva immagine trasformata
        aug_img_path = os.path.join(output_img_dir, f"aug_{i}_{filename}")
        cv2.imwrite(aug_img_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

        # Salva etichette trasformate
        aug_label_path = os.path.join(output_label_dir, f"aug_{i}_{filename.replace('.jpg', '.txt')}")
        with open(aug_label_path, "w") as f:
            for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

# Multiprocessing per velocizzare l'elaborazione
if __name__ == "__main__":
    num_workers = min(cpu_count(), 8)  # Usa max 8 core per non sovraccaricare il Mac
    print(f"Usando {num_workers} processi per l'augmentazione...")
    
    with Pool(num_workers) as p:
        p.map(process_image, image_paths)

print(f"Immagini aumentate salvate in {output_img_dir}")
print(f"Etichette aggiornate salvate in {output_label_dir}")
print(f"Numero totale di immagini aumentate: {num_original_images * augmentation_factor}")
