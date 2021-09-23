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

### 今後の予定
- jax かpytorch で自動微分・コスト関数の追加
