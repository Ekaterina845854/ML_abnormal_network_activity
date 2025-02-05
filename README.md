# ML_abnormal_network_activity
Программный макет, предназначенный для обнаружения аномальной сетевой активности на хосте с использованием методов машинного обучения.

## Описание программного макета
/project

├── capture.py

├── model.py

├── test.py

├── requirements.txt

├── Dockerfile

├── traffic_logs/   

│   ├── traffic_capture_20250201_095200.pcap

│   ├── traffic_data_20250202_105801.csv

├── anomalies.json

├── anomalies_plot.png

├── autoencoder_model.pth

Программа состоит из 4 основных файлов:

1. capture.py – Сбор сетевого трафика и его обработка
2. model.py – Моделирование и обучение автоэнкодера
3. test.py – Основной скрипт для запуска обнаружения аномалий в режиме реального времени
4. Dockerfile – Настроен для создания контейнера и установки зависимостей

### Установка

1. Клонируйте репозиторий:
   bash
   git clone https://github.com/Ekaterina845854/ML_abnormal_network_activity.git
   
2. Установите зависимости:
   bash
   pip install -r requirements.txt

### Использование

1. Запуск готовой модели для анализа сетевого трафика в режиме реального времени

  Запустите скрипт:
  bash
  python test.py --live --threshold 1.45353392
  
Где:
--live – включает режим непрерывного сбора трафика.
--threshold – порог для выявления аномалий (по умолчанию 1.45353392, вы можете установить свой).

2. Обработка пользовательского датасета: Если у вас есть собственный датасет, вы можете провести его обработку, передав путь к файлу:
  Запустите скрипт:
  bash
  python test.py --dataset /path/to/your/dataset.csv --threshold 1.45353392

Где:
--dataset – путь к CSV файлу, содержащему данные.
--threshold – порог для аномалий.

3. Запуск системы через Docker
   1) Постройте Docker образ
      Запустите скрипт:
      bash
      docker build -t anomaly-detection .

  2) Запустите контейнер
      Запустите скрипт
      bash
      docker run anomaly-detection

### Интерпретация результатов

1. Аномалии: Аномалии выявляются на основе ошибки восстановления. Пакеты, для которых ошибка восстановления превышает заданный порог, считаются аномальными. В реальном времени такие аномалии будут отображаться в логе и сохранены в файл.

2. График аномалий: Результаты аномалий можно увидеть на графике, который строится с помощью matplotlib. На графике будет показана ошибка восстановления для каждого пакета, а также пороговое значение, выше которого пакеты считаются аномальными.

3. Логи: Все обнаруженные аномалии будут записываться в файл anomalies.log, что позволит вам отслеживать события и детально анализировать поведение системы.

4. Файл аномалий: Аномалии сохраняются в файле anomalies.json, где для каждого аномального пакета указаны индекс, ошибка восстановления и данные пакета.

5. График: График ошибок восстановления (reconstruction error) и порога аномалий будет сохранен в файл anomalies_plot.png.
