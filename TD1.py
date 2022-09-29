import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self,sizes,fcts):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        stack = [nn.Linear(sizes[i//2], sizes[i//2+1]) if i%2==0 \
        else fcts[i//2 + 1] for i in range(2*len(sizes)-3)]
        self.stack = nn.Sequential(*stack)
        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

class CustomSine(nn.Module):
    def forward(self, z):
        return torch.sin(z)


def test(model, data, target):
    x=torch.FloatTensor(data)
    y=np.squeeze(model(x).data.numpy())
    return np.sqrt(np.mean((np.squeeze(target)-y)**2))


def train(model, train_data, train_target, test_data, test_target, n_epochs=200):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    e, a, l = [], [], []
    idx = np.arange(len(train_data))
    idt = np.arange(len(test_data))

    for epoch in range(n_epochs):
        optim.zero_grad()
        
        np.random.shuffle(idx)
        x = torch.FloatTensor(train_data[idx])
        target = torch.FloatTensor(train_target[idx])
        pred = model(x)

        # loss = sum([criterion(p,t) for p, t in zip(pred, target)])
        loss = criterion(pred, target)
        np.random.shuffle(idt)
        acc = test(model, test_data[idt], test_target[idt])
        e.append(epoch) 
        a.append(acc)
        l.append(loss.item())
        loss.backward()
        optim.step()
    return e, a, l

def genData(nTrainingSamples, nTestSamples):

    x = 6 * (np.random.rand(nTrainingSamples, 2)-.5)
    y = np.cos(x[:,0]) + np.sin(x[:,1])
    y = np.expand_dims(y, axis=1)

    xt = 6 * (np.random.rand(nTestSamples, 2)-.5)
    yt = np.cos(xt[:,0]) + np.sin(xt[:,1])
    yt = np.expand_dims(yt, axis=1)

    return x,y,xt,yt


if __name__ == "__main__":
    x,y,xt,yt=genData(100, 100)
    xp = np.array([[0, (u/100 - .5)*6] for u in range(100)])
    yp = np.cos(xp[:,0]) + np.sin(xp[:,1])
    
    fcts = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]
    fctsNames = ["ReLU", "Tanh", "Sigmoid"]

    fig1, ax1 = plt.subplots(3, 3)
    fig1.suptitle("Accuracy in the range [-3, 3]")
    fig2, ax2 = plt.subplots(3, 3)
    fig2.suptitle("Behaviour during training (200 epochs)")
    for i in range(3):
        for j in range(3):
            nhid = (j+1)*30
            model = Net([2, nhid, 1], \
                        [fcts[i]]*3) 
            e,acc,loss = train(model,x, y,xt,yt)

            
            pred = np.squeeze(model(torch.FloatTensor(xp)).data.numpy())

            ax1[i, j].plot(xp[:,1],yp,'b',label='Ground truth')
            ax1[i, j].plot(xp[:,1],pred,'r',label='Prediction')
            
            ax2[i, j].plot(e,acc,'b-',label='MSE')
            ax2[i, j].plot(e,loss,'r-',label='loss')

            ax1[i, j].set_title(fctsNames[i] + f" : {nhid} hidden neurons")
            ax1[i, j].legend()
            ax2[i, j].set_title(fctsNames[i] + f" : {nhid} hidden neurons")
            ax2[i, j].legend()
    

    
    fig3, ax3 = plt.subplots(1, 2)
    fig3.suptitle("Behaviour with sine activations")
    model = Net([2, 5, 1], \
                [nn.ReLU(), CustomSine()]) 
    e,acc,loss = train(model,x, y,xt,yt,n_epochs=2000)
    pred = np.squeeze(model(torch.FloatTensor(xp)).data.numpy())
    ax3[0].plot(xp[:,1],yp,'b',label='Ground truth')
    ax3[0].plot(xp[:,1],pred,'r',label='Prediction')
    ax3[1].plot(e,acc,'b-',label='MSE')
    ax3[1].plot(e,loss,'r-',label='loss')
    ax3[0].legend()
    ax3[0].set_title("Accuracy")
    ax3[1].legend()
    ax3[0].set_title("Training progress")

    plt.show()
