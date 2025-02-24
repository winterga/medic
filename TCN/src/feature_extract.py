# Greyson's implementation of (pre-trained) ResNet 50 CNN for feature extraction
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision.models as models

def load_extractor():
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
    feature_extractor.eval()
    return feature_extractor

# input: frames [batch_size, channels, height, width], model [feature_extractor]
def extract_features(frames, model):
    with torch.no_grad(): # disables gradient calculation since we ar not training the model
        features = model(frames)
    return features.squeeze() # [batch_size, 2048] tensor


# Test to make sure the feature extraction works
if __name__ == "__main__":
    model = load_extractor()
    frames = torch.randn(8, 3, 224, 224)
    features = extract_features(frames, model)
    print(features.shape) # torch.Size([8, 2048])