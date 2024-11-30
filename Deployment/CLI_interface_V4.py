# imports
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  #to improve the aestetics of the terminal while execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   #to improve the aestetics of the terminal while execution

import logging
import torch
import numpy as np
import torchvision.transforms as transforms
import warnings
from torchvision.models import vgg16, resnet18
from PIL import Image
from timm import create_model
from tensorflow.keras.models import load_model # type: ignore
import xgboost as xgb
import torch.nn as nn

warnings.filterwarnings("ignore")    #to improve the aestetics of the terminal while execution
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)



classes = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# model locations
model_paths = {
    "custom_pth": r"saved model\EnhancedDRDetectionModel.pth",
    "xgboost_json": r"saved model\XGB_model.json",
    "keras_h5": r"saved model\enhanced_dr_model.keras",
}

# Architecture of the model
class EnhancedDRDetectionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedDRDetectionModel, self).__init__()

        # VGG
        self.vgg_features = nn.Sequential(*list(vgg16(pretrained=True).features)[:10])

        # ResNet
        self.resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Swin Transformer
        self.swin_transformer = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)

        # Final classification layer
        self.fc = nn.Linear(401408 + 512 + 1024, num_classes)  # Adjusted based on feature sizes

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # VGG features
        vgg_out = self.vgg_features(x)
        vgg_out = torch.flatten(vgg_out, start_dim=1)

        # ResNet features
        resnet_out = self.resnet(x)
        resnet_out = torch.flatten(resnet_out, start_dim=1)

        # Swin Transformer featues
        swin_out = self.swin_transformer(x)

        # Concatenate features
        combined_features = torch.cat((vgg_out, resnet_out, swin_out), dim=1)

        #  Dropout and classification layer
        combined_features = self.dropout(combined_features)
        out = self.fc(combined_features)

        return out

# prediction function for the ensemble learning model
def predict_with_custom_model(image_path, model_path):
    # Load architecture
    model = EnhancedDRDetectionModel(num_classes=5)
    model.load_state_dict(torch.load(model_path))  
    model.eval()  

    # transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # loading image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0) 

    # prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    print(f"\nPrediction (Ensemble Learning Model): {classes[predicted_class]}")



# prediction function for XGBoost model
def predict_with_xgboost(image_path, model_path):
    # Load model
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)

    # transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # loading image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    input_tensor = input_tensor.flatten().numpy()  # flatten to (150528,)

    # reshape to 2D
    input_reshaped = input_tensor.reshape(1, -1)  # (1, 150528)

    # select the top 512 features
    from sklearn.feature_selection import SelectKBest
    selector = SelectKBest(k=512)
    input_selected = selector.fit_transform(input_reshaped, np.zeros((1,)))  # dummy target variable

    # conversion to Dmatrix
    dmatrix_input = xgb.DMatrix(input_selected)

    # prediction
    prediction_probabilities = xgb_model.predict(dmatrix_input)
    prediction_class_index = np.argmax(prediction_probabilities)
    # prediction_class_label = f"Class {prediction_class_index}"

    print(f"\nPrediction (XGBoost): {classes[prediction_class_index]}")



# prediction function for keras model
def predict_with_keras(image_path, model_path):
    # load model
    model = load_model(model_path)
    
    # loading image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    
    
    input_tensor = input_tensor.permute(0, 2, 3, 1).numpy()  # Shape: (1, 224, 224, 3)
    
    # prediction
    prediction = model.predict(input_tensor)
    predicted_class = prediction.argmax(axis=1)[0]
    print(f"\nPrediction (Keras Model): {classes[predicted_class]}")

# CLI Workflow
def main():
    # install_requirements()

    print("\n\n\t\t\t\tWelcome to the Diabetic Retinopathy Detection CLI")

    while True:
        # display all the .jpeg files
        image_dir = r"Deployment\images"
        jpeg_files = [f for f in os.listdir(image_dir) if f.endswith(".jpeg")]
        if not jpeg_files:
            print(f"\nNo .jpeg files found in the {image_dir} directory.")
            return
        
        print("\n\nAvailable .jpeg files:\n")
        for i, file in enumerate(jpeg_files):
            print(f"{i + 1}. {file}")
        
        # image selection
        image_index = int(input("\nSelect an image by entering the corresponding number: ")) - 1
        if image_index < 0 or image_index >= len(jpeg_files):
            print("Invalid selection.")
            return
        
        selected_image = os.path.join(image_dir, jpeg_files[image_index])
        print(f"Selected image: {selected_image}")
        
        # model selection
        print("\nAvailable Models: ")
        print("1. Ensemble Model ")
        print("2. XGBoost Model ")
        print("3. Keras Model ")

        model_choice = int(input("\nSelect a model by entering the corresponding number: "))
        
        if model_choice == 1:
            predict_with_custom_model(selected_image, model_paths["custom_pth"])
        elif model_choice == 2:
            predict_with_xgboost(selected_image, model_paths["xgboost_json"])
        elif model_choice == 3:
            predict_with_keras(selected_image, model_paths["keras_h5"])
        else:
            print("Invalid selection.")
            
        # process loop
        response = input("\n\nDo you want to exit? (yes/no): ")
        if response.lower() == "yes":
            break

if __name__ == "__main__":
    main()