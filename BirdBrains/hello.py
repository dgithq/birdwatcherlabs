from flask import Flask
from flask import render_template, request, redirect, flash, url_for

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os
from PIL import Image
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd

# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 30, 3, 1) # This convolutional layer has 3 input channels, 30 output channels
        self.conv2 = nn.Conv2d(30, 60, 3, 1) # They have 3*3 kernel and stride of size 1
        self.conv3 = nn.Conv2d(60, 130, 3, 1)
        self.conv4 = nn.Conv2d(130, 240, 3, 1)
        self.drop_out = nn.Dropout2d(p=0.2) # one drop out layer to prevent the model from being overcomplex
        self.fc1 = nn.Linear(12*12*240, 200) # fully connected layers to do the classification
        self.fc2 = nn.Linear(200,148)
        self.fc3 = nn.Linear(148, 84)
        self.fc4 = nn.Linear(84,21) #  #output features is 21, which corresponds to 21 bird species

    def forward(self, X):
        X = F.relu(self.conv1(X)) # ReLu activation function is used here
        X = F.max_pool2d(X, 2, 2) # use max pooling layer
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = self.drop_out(X)
        X = X.view(-1, 12*12*240)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1) # use softmax function to perform the classification

CNNmodel=torch.load('ResNet18modelone.pt')
class_names =['ALBATROSS', 'AMERICAN GOLDFINCH', 'AMERICAN KESTREL', 'BALTIMORE ORIOLE', 'BIRD OF PARADISE', 'BLUE HERON', 'CALIFORNIA CONDOR', 'CALIFORNIA QUAIL', 'EMPEROR PENGUIN', 'GOLDEN PHEASANT', 'HOUSE SPARROW', 'HYACINTH MACAW', 'MANDRIN DUCK', 'NORTHERN CARDINAL', 'OSTRICH', 'ROBIN', 'RUBY THROATED HUMMINGBIRD', 'SCARLET IBIS', 'SPOONBILL', 'TOUCHAN', 'YELLOW HEADED BLACKBIRD']
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key='12345'

def hello(name=None):
    return render_template('index.html', name=name)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        
        if 'file' not in request.files:
            return render_template('index.html')
        
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if not file:
            return render_template('index.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save('./uploads/'+filename)
            ans = classify(filename)
            return ans
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')



def classify(filename = None):
    image = Image.open('./uploads/'+filename)
    loader = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    image = loader(image)
    #imshow(image)
    image = image.unsqueeze(0)
    output = CNNmodel(image)
    _, preds = torch.max(output, 1)
    answer = class_names[preds[0]]
    os.remove('./uploads/'+ filename)
    return answer

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS