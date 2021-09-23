from qulacs import QuantumState, ParametricQuantumCircuit, Observable
import math

def gradient_by_half_pi(index_in_parametrized_gates, pqc, obs_list):
    '''
        index_in_parametrized_gates: the gate index in the parametrized gates
        pqc: parametrized quantum circuit
        obs_list: the list of observables

        This function calculates the gradients of RX, RY, and RZ gates for given observables.
    '''
    n_qubits = pqc.get_qubit_count()
    n_gates = pqc.get_gate_count()
    if index_in_parametrized_gates<0 or index_in_parametrized_gates>=pqc.get_parameter_count():
        print("the number of parametrized gates is ", pqc.get_parameter_count())
        return
    gate_index = pqc.get_parametric_gate_position(index_in_parametrized_gates)
    # get_parameter, set_parameter はパラメータ付き量子ゲートの中でのindex 
    angle = pqc.get_parameter(index_in_parametrized_gates)
    pqc.set_parameter(index_in_parametrized_gates, angle + math.pi/2)
    state = QuantumState(n_qubits)
    for i in range(n_gates):
        if i==gate_index:
            copy_state = state.copy()
        gate_i = pqc.get_gate(i)
        gate_i.update_quantum_state(state)
    exp_val_plus_list = [ obs.get_expectation_value(state) for obs in obs_list]
    qc.set_parameter(index_in_parametrized_gates, angle - math.pi/2)
    for i in range(gate_index, n_gates):
        gate_i = pqc.get_gate(i)
        gate_i.update_quantum_state(copy_state)
    exp_val_minus_list = [ obs.get_expectation_value(copy_state) for obs in obs_list]
    
    qc.set_parameter(index_in_parametrized_gates, angle)
    grad_list = [ (p-m)/2 for p, m in zip(exp_val_plus_list, exp_val_minus_list) ]
    return grad_list

# calculating gradients of CRX, CRY, and CRZ
def gradient_by_four_terms(index_in_parametrized_gates, pqc, obs_list):
    '''
        index_in_parametrized_gates: the gate index in the parametrized gates
        pqc: parametrized quantum circuit
        obs_list: the list of observables

        This function calculates the gradients of CRX, CRY, and CRZ gates for given observables.
    '''
    n_qubits = pqc.get_qubit_count()
    n_gates = pqc.get_gate_count()
    if index_in_parametrized_gates<0 or index_in_parametrized_gates>=pqc.get_parameter_count():
        print("the number of parametrized gates is ", pqc.get_parameter_count())
        return
    gate_index = pqc.get_parametric_gate_position(index_in_parametrized_gates)
    angle = pqc.get_parameter(index_in_parametrized_gates)
    # +pi/2
    pqc.set_parameter(index_in_parametrized_gates, angle + math.pi/2)            
    state = QuantumState(n_qubits)
    for i in range(n_gates):
        if i==gate_index:
            copy_state = state.copy()
        gate_i = pqc.get_gate(i)
        gate_i.update_quantum_state(state)
    exp_val_plus1_list = [ obs.get_expectation_value(state) for obs in obs_list]
    # -pi/2
    state = copy_state.copy()
    qc.set_parameter(index_in_parametrized_gates, angle - math.pi/2)
    for i in range(gate_index, n_gates):
        gate_i = pqc.get_gate(i)
        gate_i.update_quantum_state(state)
    exp_val_minus1_list = [ obs.get_expectation_value(state) for obs in obs_list]
    # +1.5pi
    state = copy_state.copy()
    qc.set_parameter(index_in_parametrized_gates, angle + math.pi * 1.5)
    for i in range(gate_index, n_gates):
        gate_i = pqc.get_gate(i)
        gate_i.update_quantum_state(state)
    exp_val_plus2_list = [ obs.get_expectation_value(state) for obs in obs_list]
    # -1.5pi
    qc.set_parameter(index_in_parametrized_gates, angle - math.pi * 1.5)
    for i in range(gate_index, n_gates):
        gate_i = pqc.get_gate(i)
        gate_i.update_quantum_state(copy_state)
    exp_val_minus2_list = [ obs.get_expectation_value(copy_state) for obs in obs_list]
    
    qc.set_parameter(index_in_parametrized_gates, angle)
    # calculate gradient
    const_p = (math.sqrt(2)+1)/(4*math.sqrt(2))
    const_m = (math.sqrt(2)-1)/(4*math.sqrt(2))
    grad_list = [
        const_p*(exp_val_plus1_list[i]-exp_val_minus1_list[i]) \
        - const_m*(exp_val_plus2_list[i]-exp_val_minus2_list[i])
        for i in range(len(obs_list))
        ]
    return grad_list


# 各パラメータに対して、各オブザーバブルに対する勾配
# 各行は、あるパラメータにおいて各オブザーバブルに対する勾配
def get_gradient(pqc, obs_list):
    '''
        pqc: parametrized quantum circuit
        obs_list: the list of observables
    '''
    n_parametrized_gate = pqc.get_parameter_count()
    grad_list = []
    for i in range(n_parametrized_gate):
        gate_index = pqc.get_parametric_gate_position(i)
        control_qubit_list = qc.get_gate(gate_index).get_control_index_list()
        if len(control_qubit_list)==0: # RX, RY, RZ
            tmp_grad = gradient_by_half_pi(i, pqc, obs_list)
        elif len(control_qubit_list)==1: # CRX, CRY, CRZ
            tmp_grad = gradient_by_four_terms(i, pqc, obs_list)
        else:
            print("the number of control gate is greater than 1.")
            print("we cannnot calculate the gradient of the gate.")
            break
        grad_list.append(tmp_grad)
    return grad_list


if __name__=='__main__':
    import pandas as pd
    import numpy as np
    n_loops = 5
    params = np.random.randn(6*n_loops)
    # params = [0.01*math.pi*i for i in range(6)]
    n_qubits = 3
    state = QuantumState(n_qubits)
    qc = ParametricQuantumCircuit(n_qubits)
    for i in range(n_loops):
        qc.add_X_gate(1)
        qc.add_parametric_CRX_gate(0, 2, -params[i*6+0])
        qc.add_parametric_CRY_gate(1, 0, -params[i*6+1])
        qc.add_parametric_CRZ_gate(2, 1, -params[i*6+2])
        qc.add_CNOT_gate(0,1)
        qc.add_CNOT_gate(1,2)
        qc.add_parametric_RX_gate(2, -params[i*6+3])
        qc.add_parametric_RY_gate(1, -params[i*6+4])
        qc.add_parametric_RZ_gate(0, -params[i*6+5])
    observable_list = []
    for i in range(n_qubits):
        observable = Observable(n_qubits)
        observable.add_operator( 1.0, "Z "+str(i) )
        observable_list.append(observable)
    grad_qul = get_gradient(qc, observable_list)
    
    grad_qul = pd.DataFrame( grad_qul )
    print( grad_qul )
        
    import pennylane as qml
    from pennylane import numpy as pnp
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(params):
        for i in range(n_loops):
            qml.PauliX(wires=n_qubits-2)
            qml.CRX(params[i*6+0], wires=[n_qubits-1, n_qubits-3])
            qml.CRY(params[i*6+1], wires=[n_qubits-2, n_qubits-1])
            qml.CRZ(params[i*6+2], wires=[n_qubits-3, n_qubits-2])
            qml.CNOT(wires=[n_qubits-1,n_qubits-2])
            qml.CNOT(wires=[n_qubits-2,n_qubits-3])
            qml.RX(params[i*6+3], wires=n_qubits-3)
            qml.RY(params[i*6+4], wires=n_qubits-2)
            qml.RZ(params[i*6+5], wires=n_qubits-1)
        return [qml.expval(qml.PauliZ(n_qubits-1-i)) for i in range(n_qubits)]
    params_pnp = pnp.array(params, requires_grad=True)
    grad_pen = qml.jacobian(circuit)(params_pnp)
    grad_pen = pd.DataFrame( grad_pen )
    print( grad_pen )
    print( np.isclose(grad_qul.to_numpy(), -1*grad_pen.T.to_numpy()) )
    print( np.allclose(grad_qul.to_numpy(), -1*grad_pen.T.to_numpy()) )
    