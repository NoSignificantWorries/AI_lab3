import cv2
import os

def visualize_yolo_annotations(image_path, annotation_path, output_path):
    """Визуализирует YOLO аннотации на изображении."""

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return

    height, width = img.shape[:2]

    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

                # Преобразуем нормализованные координаты в пиксельные
                x_center = int(x_center * width)
                y_center = int(y_center * height)
                bbox_width = int(bbox_width * width)
                bbox_height = int(bbox_height * height)

                # Вычисляем координаты верхнего левого угла bounding box
                x1 = int(x_center - bbox_width / 2)
                y1 = int(y_center - bbox_height / 2)
                x2 = int(x_center + bbox_width / 2)
                y2 = int(y_center + bbox_height / 2)

                # Рисуем bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зеленый цвет

                # Пишем класс объекта
                cv2.putText(img, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(output_path, img)
        print(f"Визуализация сохранена в {output_path}")

    except FileNotFoundError:
        print(f"Ошибка: Файл аннотаций {annotation_path} не найден")
    except Exception as e:
        print(f"Ошибка при обработке файла {annotation_path}: {e}")

# Пример использования:
image_dir = "data/dataset/images/val"
annotation_dir = "data/dataset/labels/val"
output_dir = "data/test"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(annotation_dir):
    if filename.endswith(".txt"):
        image_name = filename[:-4] + ".jpg"  # Или .png, в зависимости от формата
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, filename)
        output_path = os.path.join(output_dir, "visualized_" + filename[:-4] + ".jpg")
        visualize_yolo_annotations(image_path, annotation_path, output_path)
