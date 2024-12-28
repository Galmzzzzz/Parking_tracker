import cv2
import pickle
import cvzone
import numpy as np

# Чтение видео с парковкой
cap = cv2.VideoCapture('carPark.mp4')

# Загрузка списка координат парковочных мест из файла
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Ширина и высота прямоугольников, соответствующих парковочным местам
width, height = 107, 48

# Функция для проверки состояния парковочных мест
def checkParkingSpace(imgPro):
    spaceCounter = 0  # Счетчик свободных мест

    for pos in posList:
        x, y = pos  # Координаты парковочного места

        # Вырезаем область изображения, соответствующую парковочному месту
        imgCrop = imgPro[y:y + height, x:x + width]

        # Считаем количество белых пикселей (занятость)
        count = cv2.countNonZero(imgCrop)

        # Условие для определения, свободно место или занято
        if count < 900:  # Если белых пикселей меньше 900, место считается свободным
            color = (0, 255, 0)  # Зеленый цвет для свободного места
            thickness = 5  # Толщина рамки
            spaceCounter += 1  # Увеличиваем счетчик свободных мест
        else:
            color = (0, 0, 255)  # Красный цвет для занятого места
            thickness = 2  # Толщина рамки

        # Рисуем прямоугольник на исходном изображении
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        # Отображаем количество пикселей в рамке
        # cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
        #                    thickness=2, offset=0, colorR=color)

    # Выводим общее количество свободных мест
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0,200,0))

# Основной цикл обработки видео
while True:
    # Проверяем, если видео дошло до конца, то начинаем заново
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Считываем текущий кадр
    success, img = cap.read()

    # Преобразование изображения в оттенки серого
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применение размытия для уменьшения шума
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    # Адаптивная пороговая обработка для выделения объектов
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)

    # Применение медианного фильтра для сглаживания изображения
    imgMedian = cv2.medianBlur(imgThreshold, 5)

    # Увеличение объектов на изображении (дилатация)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Проверка парковочных мест на текущем кадре
    checkParkingSpace(imgDilate)

    # Отображение исходного изображения с наложенными рамками
    cv2.imshow("Image", img)

    # Ожидание 10 миллисекунд перед переходом к следующему кадру
    cv2.waitKey(10)
