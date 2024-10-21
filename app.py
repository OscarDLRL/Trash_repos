import io
import json
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

# Cargar el archivo imagenet_class_index.json
imagenet_class_index = json.load(open("C:/Users/oscar/Downloads/imagenet_class_index.json"))

# Cargar el modelo preentrenado
model = models.densenet121(weights='IMAGENET1K_V1')
model.eval()  # Configurar el modelo en modo de evaluación

# Transformar la imagen para que sea compatible con el modelo
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],  # Valores medios del conjunto de datos
                                            [0.229, 0.224, 0.225])])  # Desviación estándar
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# Obtener la predicción de la imagen
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())  # Convertir el valor de tensor a un índice
    return imagenet_class_index[predicted_idx]

# Ruta para procesar las imágenes subidas
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener el archivo de la solicitud POST
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

# Iniciar el servidor Flask
if __name__ == '__main__':
    app.run()
    
# imagen a ver
#C:\Users\oscar\OneDrive\Escritorio\plastic_1748.png