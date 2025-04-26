import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )
        
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),  # 256 features confirmed
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            
            nn.Linear(in_features=84, out_features=1)
        )
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_model(x)
        x = F.sigmoid(x)
        
        return x

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load('model/model_weights.pth', map_location=device))
    model.eval()
    return model, device

def preprocess_image(image_data):
    # Define the transformations with the correct size of 106x106
    transform = transforms.Compose([
        transforms.Resize((106, 106)),  # Exact size needed to produce 256 features
        transforms.ToTensor(),
    ])
    
    # Open image from binary data
    img = Image.open(io.BytesIO(image_data))
    
    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor

def get_prediction(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get prediction (0 for healthy, 1 for tumor)
    prediction = 'Tumour' if output.item() > 0.5 else 'Healthy'
    confidence = output.item() if output.item() > 0.5 else 1 - output.item()
    
    return prediction, confidence * 100  # Return confidence as percentage