import torch
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

datos = pd.read_csv('Churn_Modelling.csv')
display(datos.head())

#separamos la ultima colmna para que se varible destino
datos_y = datos[datos.columns[-1]]

#Elminanos las columnas que no funcionaran
datos_x = datos.drop(columns=['RowNumber','CustomerId','Surname','Exited'])

#hacemos un one hot encoding, ya que las redes reuronales no procesan data de str
datos_x = pd.get_dummies(datos_x)

#Ahora vamos a escalar los valores para que se encuentre dentro de un rango mas corto
escalador = StandardScaler()
datos_x = escalador.fit_transform(datos_x)

print('resgitros {}'.format(datos_x.shape[0]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(datos_x, datos_y, test_size=0.2,random_state=2)

print('X_train:{}, X_test:{}, y_train:{}, y_test:{}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

n_entradas = X_train.shape[1]


t_X_train = torch.from_numpy(X_train).float().to('cuda')
t_X_test = torch.from_numpy(X_test).float().to('cuda')

t_y_train = torch.from_numpy(y_train.values).float().to('cuda')
t_y_test = torch.from_numpy(y_test.values).float().to('cuda')

t_y_train = t_y_train[:,None]
t_y_test = t_y_test[:,None]

test = TensorDataset(t_x_test,t_y_test)
print(test[0])

#NN
class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8, 1)
    
    def forward(self, inputs):
        pred_1 = torch.sigmoid(input=self.linear1(inputs))
        pred_2 = torch.sigmoid(input=self.linear2(pred_1))
        pred_f = torch.sigmoid(input=self.linear3(pred_2))
        return pred_f

#training epochs
lr = 0.001
epochs = 2000
estatus_print = 100

model = Red(n_entradas=n_entradas) # si quieres entrenar con cuda sino lo quitas
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
print("Arquitectura del modelo: {}".format(model))
historico = pd.DataFrame()

print("Entranando el modelo")
for epoch in range(1, epochs+1):
    y_pred= model(t_X_train)
    loss = loss_fn(input=y_pred, target=t_y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % estatus_print == 0:
        print(f"\nEpoch {epoch} \t Loss: {round(loss.item(), 4)}")
    
    with torch.no_grad():
        y_pred = model(t_X_test)
        y_pred_class = y_pred.round()
        correct = (y_pred_class == t_y_test).sum()
        accuracy = 100 * correct / float(len(t_y_test))
        if epoch % estatus_print == 0:
            print("Accuracy: {}".format(accuracy.item()))
    
    df_tmp = pd.DataFrame(data={
        'Epoch': epoch,
        'Loss': round(loss.item(), 4),
        'Accuracy': round(accuracy.item(), 4)
    }, index=[0])
    historico = pd.concat(objs=[historico, df_tmp], ignore_index=True, sort=False)

print("Accuracy final: {}".format(round(accuracy.item(), 4)))