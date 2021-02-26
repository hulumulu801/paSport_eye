# Как установить?(ТОЛЬКО LINUX):
**Внимание: Путь не должен содержать русских букв!!**

**Рекомендую использовать Virtualenv**

* открываем терминал и вставляем следующее содержимое:
  - обновляем систему:
  
        sudo apt-get update -y
        sudo apt-get upgrade -y
        sudo apt-get dist-upgrade -y
  - устанавливаем git:
  
        sudo apt install git -y
  - готовим машину к установке dlib:
  
        sudo apt-get install build-essential cmake -y
        sudo apt-get install libopenblas-dev liblapack-dev -y
        sudo apt-get install libx11-dev libgtk-3-dev -y
        sudo apt-get install python3 python3-dev python3-pip -y
        sudo apt-get install cmake -y
        sudo apt-get install python3-matplotlib python3-numpy python3-pil python3-scipy -y
  - устанавливаем tesseract-ocr:
  
        sudo apt-get install tesseract-ocr -y
        sudo apt-get install tesseract-ocr-all -y
  - скачиваем репозиторий:
  
        git clone https://github.com/hulumulu801/pasport_eye.git
  - переходим в директорию:
  
        cd pasport_eye/
  - еще раз:
  
        cd pasport_eye_v_0.1.1/
  - скачиваем shape_predictor_68_face_landmarks.dat:
  
        wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
        bunzip2 shape_predictor_68_face_landmarks.dat.bz2
  - устанавливаем нужные библиотеки:
  
        pip3 install -r requirements.txt
# Использование:

  - help:
  
        python3 pasport_eye.py --help
  - создается папка с названием базы + _recogn, в ней подпапка pict - лица из паспортов и подпапка descript - дескрипторы в двоичном формате, выхлоп распознования - в терминал:
  
        python3 pasport_eye.py -i /ПУТЬ/К/ПАПКЕ/С/ИЗОБРАЖЕНИЯМИ -m 1
  - дескрипторы(в формате: float), изображение лиц(в формате: base64), а так же данные паспортов сохраняются в НАЗВАНИЕ_БАЗЫ_С_ИЗОБРАЖЕНИЯМИ.csv:
  
        python3 pasport_eye.py -i /ПУТЬ/К/ПАПКЕ/С/ИЗОБРАЖЕНИЯМИ -m 2
  - ВКЛЮЧИТЬ метод улучшения распознавания "нечитаемых зон":

          python3 pasport_eye.py -i /ПУТЬ/К/ПАПКЕ/С/ИЗОБРАЖЕНИЯМИ -m 1 -i_r ON
          
  - ОТКЛЮЧИТЬ метод улучшения распознавания "нечитаемых зон":

          
          python3 pasport_eye.py -i /ПУТЬ/К/ПАПКЕ/С/ИЗОБРАЖЕНИЯМИ -m 1 -i_r OFF
