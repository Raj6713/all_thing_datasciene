import torch
import torch.nn as nn
import torch.optim as optim

# # 1. Create dummy dataset
# X = torch.randn(100, 2)   # 100 samples, 2 features
# y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)  # label = 1 if x0+x1 > 0 else 0
X = torch.randn(100,2)
Y = (X[:,0]+X[:,1]>0).float().unsqueeze(1)



# # 2. Define model
# class LogisticRegression(nn.Module):
#     def __init__(self, input_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = nn.Linear(input_dim, 1)  # one output neuron

#     def forward(self, x):
#         return torch.sigmoid(self.linear(x))  # probability between 0 and 1
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# # 3. Initialize model, loss, optimizer
# model = LogisticRegression(input_dim=2)
# criterion = nn.BCELoss()                  # Binary Cross Entropy Loss
# optimizer = optim.SGD(model.parameters(), lr=0.1)

model = LogisticRegression(input_dim=2)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# # 4. Training loop
# epochs = 100
# for epoch in range(epochs):
#     # Forward pass
#     y_pred = model(X)
#     loss = criterion(y_pred, y)

#     # Backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 10 == 0:
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# # 5. Evaluation
# with torch.no_grad():
#     preds = model(X)
#     predicted_classes = (preds >= 0.5).float()
#     accuracy = (predicted_classes.eq(y).sum() / y.shape[0]).item()
#     print(f"Accuracy: {accuracy*100:.2f}%")
epochs = 100
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) %10 == 0:
        print(f"Epochs {epoch+1}/{epochs}, Loss: {loss.item(): .4f}")
with torch.no_grad():
    preds = model(X)
    predicted_classes = (preds >= 0.5).float()
    accuracy = (predicted_classes.eq(Y).sum()/Y.shape[0]).item()
    print(f"Accuracy: {accuracy*100:.2f}")
    