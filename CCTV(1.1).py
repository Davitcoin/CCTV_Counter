import cv2

# Cargar el clasificador pre-entrenado para la detección de personas en la parte superior del cuerpo
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Inicializar el contador de personas
person_count = 0

# Capturar el video desde una fuente (por ejemplo, una cámara o archivo de video)
video_capture = cv2.VideoCapture(0)

while True:
    # Leer el cuadro actual del video
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convertir el cuadro a escala de grises para la detección de personas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar personas en la parte superior del cuerpo utilizando el clasificador
    upperbodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de las personas detectadas y contarlas
    for (x, y, w, h) in upperbodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        person_count += 1

    # Mostrar el número de personas detectadas en tiempo real
    cv2.putText(frame, "Personas: {}".format(person_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar el cuadro con las personas detectadas y el conteo
    cv2.imshow('CCTV', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
video_capture.release()
cv2.destroyAllWindows()
