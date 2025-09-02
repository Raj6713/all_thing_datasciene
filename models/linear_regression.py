# import torch
# import torch.nn as nn
# import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)
X = torch.unsqueeze(torch.linspace(0, 10,100), dim=1)
Y = 2 * X + 3 + torch.randn(X.size())*0.5


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1,1)
        self.linear2 = nn.Linear(1,1)
    
    def forward(self, x):
        x= self.linear1(x)
        # x- self.linear2(x)
        return x

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)


epochs = 200
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%20 == 0:
        w, b = model.linear.weight.item(), model.linear.bias.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss:{loss.item():.4f}, w: {w:.2f}, b:{b:.2f}")
test_x = torch.tensor([[4.0]])
predicted = model(test_x).item()
print(f"\nPrediction for x=4 {predicted:.2f}")
