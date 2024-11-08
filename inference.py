import torch
import mlflow
import mlflow.pytorch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("http://127.0.0.1:8081/")

def preprocess_input(data):
    """
    Preprocess the input data for XOR model.
    Input: List or numpy array of shape (n, 2)
    Output: Torch tensor of shape (n, 2)
    """
    if isinstance(data, list):
        data = np.array(data)

    assert data.shape[1] == 2, "Input data should have 2 features (e.g., [[0, 1], [1, 0]])"

    tensor_data = torch.tensor(data, dtype=torch.float32, device=device)
    return tensor_data

def predict(data):
    print("Loading the model from MLflow...")
    model_uri = "models:/xor/latest"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)
    model.eval()

    inputs = preprocess_input(data)

    with torch.no_grad():
        outputs = model(inputs)
        print(outputs)
        predictions = (outputs >= 0.5).float().cpu().numpy()  # Binarize output (0 or 1)

    return predictions

if __name__ == '__main__':
    test_data = [[0, 0], [0, 1], [1, 0], [1, 1]]

    predictions = predict(test_data)

    for i, input_data in enumerate(test_data):
        print(f"Input: {input_data} -> Prediction: {int(predictions[i][0])}")
