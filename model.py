import mlflow
import torch
assert torch.version.__version__ >= '1.0.0'

import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("http://127.0.0.1:8081/")

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(2, 4, True)
        self.fc2 = nn.Linear(4, 1, True)

    def forward(self, x):
        h = F.tanh(self.fc1(x))
        y = F.sigmoid(self.fc2(h))
        return y

def train(model, loss_fn, optimizer):
    model = model.to(device)

    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device).float()
    y = torch.tensor([[0], [1], [1], [0]], device=device).float()

    model.train()

    pred = model(X)
    _loss = loss_fn(pred, y)
    _loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return _loss

if __name__ == '__main__':
    epochs = 5000
    loss_fn = nn.BCELoss()
    model = XOR().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": 1e-3,
            "loss_function": loss_fn.__class__.__name__,
            "optimizer": "SGD",
        }
        mlflow.log_params(params)

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
            mlflow.log_artifact("model_summary.txt")

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            loss = train( model, loss_fn, optimizer)
            mlflow.log_metric(key="train_loss", value=loss, step=epoch)
            if (epoch % 100 == 0):
                print("Epoch: {0}, Loss: {1}, ".format(epoch, loss.to("cpu").detach().numpy()))

        mlflow.pytorch.log_model(model, registered_model_name="xor", artifact_path="xor-model")
