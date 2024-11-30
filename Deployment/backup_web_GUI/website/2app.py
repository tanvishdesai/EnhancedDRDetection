# imports
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet18
from timm import create_model
import xgboost as xgb
from tensorflow import keras
import os
import cv2
from sklearn.feature_selection import SelectKBest

app = Flask(__name__)

# Architecture of the model
class EnhancedDRDetectionModel(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedDRDetectionModel, self).__init__()
        # VGG
        self.vgg_features = torch.nn.Sequential(*list(vgg16(pretrained=True).features)[:10])
        # ResNet
        self.resnet = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        # Swin Transformer
        self.swin_transformer = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        # Final classification layer
        self.fc = torch.nn.Linear(401408 + 512 + 1024, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        vgg_out = torch.flatten(self.vgg_features(x), start_dim=1)
        resnet_out = torch.flatten(self.resnet(x), start_dim=1)
        swin_out = self.swin_transformer(x)
        combined_features = torch.cat((vgg_out, resnet_out, swin_out), dim=1)
        combined_features = self.dropout(combined_features)
        out = self.fc(combined_features)
        return out

model = EnhancedDRDetectionModel()
model.load_state_dict(torch.load(r'saved model\EnhancedDRDetectionModel.pth', map_location='cpu'))
model.eval()

xgboost_model = xgb.XGBClassifier()
xgboost_model.load_model(r'saved model\XGB_model.json')

keras_model = keras.models.load_model(r'saved model\enhanced_dr_model.keras')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    return transform(image)

@app.route('/')
def index():
    return render_template('s_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or not allowed_file(request.files['image'].filename):
        return jsonify({'error': 'Invalid image file'}), 400

    image_file = request.files['image']
    model_name = request.form.get('model')

    # temp location for image
    image_path = os.path.join('Deployment', 'backup_web_GUI', 'website', 'static', 'temp.jpg')
    image_file.save(image_path)

    if model_name == 'model1': # ensemble
        image = preprocess_image(image_path).unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = int(predicted.item())

    elif model_name == 'model2':  # XGB
        image = preprocess_image(image_path).numpy()
        image = cv2.resize(image.transpose(1, 2, 0), (16, 32)).flatten()
        selector = SelectKBest(k=512)
        selected_features = selector.fit_transform(image.reshape(1, -1), np.zeros(1))
        predicted_class = int(xgboost_model.predict(selected_features)[0])

    elif model_name == 'model3':  # Keras
        image = preprocess_image(image_path).numpy().transpose((1, 2, 0))
        image = image.reshape((1, 224, 224, 3))
        predicted_class = int(np.argmax(keras_model.predict(image)))

    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)