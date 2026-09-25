import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob

MODEL_PATH = 'runs/segment/train/weights/best.pt'
SOURCE_DIR = 'dataset/valid/images'
OUTPUT_DIR = 'results_staging'
PIXEL_SPACING_MM = 0.7
CONF_THRESHOLD = 0.4

CLASS_NAMES = {0: "Adenocarcinoma", 1: "Cancer", 2: "Nodule"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 165, 255), 2: (255, 255, 0)}


def calculate_diameter_mm(mask_array, spacing):
    """Расчет эквивалентного диаметра узла в мм"""
    area_px = np.sum(mask_array > 0.5)
    if area_px == 0:
        return 0.0
    area_mm2 = area_px * (spacing ** 2)
    return round(2 * np.sqrt(area_mm2 / np.pi), 1)


def get_stage_info(class_id, diameter_mm):
    """Определение стадии риска по классу и размеру"""
    if class_id in [0, 1]:
        return f"{CLASS_NAMES[class_id]}\nRisk: HIGH", CLASS_COLORS[class_id]
    elif class_id == 2:
        if diameter_mm < 6:
            return "Nodule (<6mm)\nStage: I", (0, 255, 0)
        elif diameter_mm <= 30:
            return "Nodule (6-30mm)\nStage: II", (0, 165, 255)
        else:
            return "Nodule (>30mm)\nStage: III+", (0, 0, 255)
    return "Unknown", (128, 128, 128)


def process_image(image_path, model, out_dir):
    """Обработка одного изображения: инференс, расчет диаметра, визуализация"""
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w, _ = img.shape
    results = model.predict(source=image_path, conf=CONF_THRESHOLD, iou=0.5, verbose=False)

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for mask, box, cls, conf in zip(masks, boxes, classes, confs):
            # Масштабирование маски до оригинального размера
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Расчет диаметра и стадии
            diam_mm = calculate_diameter_mm(mask_resized, PIXEL_SPACING_MM)
            stage_text, color = get_stage_info(cls, diam_mm)

            # Наложение маски и отрисовка рамки
            colored_mask = np.zeros_like(img)
            colored_mask[mask_resized > 0.5] = color
            img = cv2.addWeighted(img, 0.75, colored_mask, 0.25, 0)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Вывод текстовой информации
            top_label = f"{CLASS_NAMES[cls]} | D={diam_mm}mm | Conf:{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(top_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            cv2.putText(img, top_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            stage_lines = stage_text.split('\n')
            for i, line in enumerate(stage_lines):
                cv2.putText(img, line, (x1, y1 + 20 + (i * 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохранение результата
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, img)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        image_paths.extend(glob(os.path.join(SOURCE_DIR, ext)))

    for img_path in image_paths:
        process_image(img_path, model, OUTPUT_DIR)


if __name__ == '__main__':
    main()