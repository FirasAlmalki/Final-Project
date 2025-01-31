from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO("yolov10n.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="signboard.yaml", epochs=300, device='cuda')

if __name__ == '__main__':  # Corrected the name check
    train_model()  # Corrected the function call
