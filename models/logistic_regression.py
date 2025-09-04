import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(100,2)
Y = (X[:,0]+X[:,1]>0).float().unsqueeze(1)



class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))



model = LogisticRegression(input_dim=2)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


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
    