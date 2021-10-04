## python wrapper
qulacs のparametric quantum circuit を拡張するためのコードを置く予定です。
現在のところ, 
https://github.com/puyokw/qulacs/tree/add_controlled_parametrized_gate
にある本家のqulacsからforkしたレポジトリに追加していくつもりです(どこかでpull request を出す予定)。
```
add_parametric_CRX_gate(control_qubit_index, target_qubit_index, angle)
add_parametric_CRY_gate(control_qubit_index, target_qubit_index, angle)
add_parametric_CRZ_gate(control_qubit_index, target_qubit_index, angle)
```
の3つのパラメータ付きのゲートを追加しました。
ここのラッパーではこれらの勾配を求める関数を提供しています。

## 使い方
### 勾配計算
```
from qulacs import QuantumState, ParametricQuantumCircuit, Observable

n_qubits = 3
state = QuantumState(n_qubits)
qc = ParametricQuantumCircuit(n_qubits)
for i in range(n_loops):
    qc.add_X_gate(1)
    qc.add_parametric_CRX_gate(0, 2, 0.1)
    qc.add_parametric_CRY_gate(1, 0, 0.2)
    qc.add_parametric_CRZ_gate(2, 1, 0.3)
    qc.add_CNOT_gate(0,1)
    qc.add_CNOT_gate(1,2)
    qc.add_parametric_RX_gate(2, 0.4)
    qc.add_parametric_RY_gate(1, 0.5)
    qc.add_parametric_RZ_gate(0, 0.6)
observable_list = []
for i in range(n_qubits):
    observable = Observable(n_qubits)
    observable.add_operator( 1.0, "Z "+str(i) )
    observable_list.append(observable)
grad = get_gradient(qc, observable_list)
print(grad)
```

### 最適化する
```
from qulacs import QuantumState, ParametricQuantumCircuit, Observable
from gradients import get_gradient
from optimizers_pytorch import PQClayer
import torch

torch.random.manual_seed(2)
n_output = 1
n_qubits = 3
state = QuantumState(n_qubits)
def parametrized_quantum_cirucit(input, params):
    qc = ParametricQuantumCircuit(n_qubits)
    for i in range(len(input)):
        qc.add_RY_gate(i, input[i])
    qc.add_X_gate(1)
    qc.add_parametric_CRX_gate(0, 2, params[0])
    qc.add_parametric_CRY_gate(1, 0, -params[1])
    qc.add_parametric_CRZ_gate(2, 1, 2*params[2])
    qc.add_CNOT_gate(0,1)
    qc.add_CNOT_gate(1,2)
    qc.add_parametric_RX_gate(2, -3*params[3])
    qc.add_parametric_RY_gate(1, params[4])
    qc.add_parametric_RZ_gate(0, 1.5*params[5])
    return qc
observable_list = []
for i in range(min(n_output, n_qubits)):
    observable = Observable(n_qubits)
    observable.add_operator( 1.0, "Z "+str(i) )
    observable_list.append(observable)
# make dataset
import functools
N = 20
n_params = 6
x = torch.randn(N, n_qubits, requires_grad=True)
params = torch.randn(n_params, requires_grad=True)
pqc_layer = PQClayer(params, parametrized_quantum_cirucit, observable_list)
model = torch.nn.Sequential(pqc_layer)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # use SGD
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # use Adam
y = torch.randn( (N, len(observable_list)) )
for t in range(5):
    optimizer.zero_grad()
    y_pred  = model(x)
    loss = torch.nn.MSELoss()
    output = loss(y, y_pred)
    output.backward()
    optimizer.step()
    print("step: ", t, ", loss: ", output.item())
```

