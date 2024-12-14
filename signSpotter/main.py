from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO("yolov10x.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="signboard.yaml", epochs=500, device='cpu')

if __name__ == '__main__':  # Corrected the name check
    train_model()  # Corrected the function call
