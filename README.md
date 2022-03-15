# Описание:
**Скрипт для распознавания: паспортов РФ, лиц из паспортов!**
# Под капотом:
python3

Ubuntu 20.04.4 LTS
# Версии:

**pasport_eye_v_0.1.0**

- Основные библиотеки:

      OpenCV, Tesseract, Dlib

- Распознавание первой страницы паспорта РФ

**pasport_eye_v_0.1.1**

- Основные библиотеки:

      OpenCV, Tesseract, Dlib
     
- Улучшена работа с файлами, надеюсь больше аварийного останова не будет)

- Добавлен новый метод(-i_r или --improved_recognition), который должен улучшить распознавание "нечитаемых зон". НО и "выхлоп" от этого метода ОЧЕНЬ БОЛЬШОЙ, а также ОЧЕНЬ ДОЛГОЕ ВЫПОЛНЕНИЕ СКРИПТА

- Произведена декомпозиция проекта

**pasport_eye_v_0.2.0**


# Какие проблемы:
**Проективное искажение изображения документа.** При съемке камерой углы и их отношения, а также пропорции объектов изменяются в зависимости от ракурса съемки. Это приводит к тому, что классические алгоритмы (поиск опорных линий, выделения текстовых полей и прочие) не могут применяться напрямую, а требуют предварительной проективной нормализации изображения.©

**Блики.** Глянцевая пленка, голограммы и прочие элементы защиты, которые помогают нам отличить настоящий паспорт от поддельного, очень сильно мешают при распознавании (частично уничтожая информацию). Попробуйте посмотрите на свой паспорт даже через объектив фотоаппарата (например, с помощью стандартного приложения камеры вашего смартфона) под разными углами, и вы сразу поймете всю глубину проблемы.©

**Неравномерность освещения.** В отличии от сканера, где используется свой осветитель, при фотографировании документа свет поступает от внешних источников неконтролируемым образом. Отсюда возникает еще ряд таких проблем, как тени и неточность передачи цвета.©

**Дефокусировка и смазывание.** Возникает из-за постоянного смещения камеры во время фото (ведь съемка идет без использования штатива).

**Границы паспорта и проективный базис.** В условиях "шума" выделить линейные границы, углы, скругления и прочие примитивы; сгенерировать и выбрать варианты границ документа, наиболее соответствующие модели. После определения проективного базиса необходимо проективно исправить зону изображения, произвести позиционирование полей.

# Обратная связь:

hulumulu801@gmail.com
# Пожертвования:

<noscript><a href="https://liberapay.com/hulumulu801/donate"><img alt="Donate using Liberapay" src="https://liberapay.com/assets/widgets/donate.svg"></a></noscript>
