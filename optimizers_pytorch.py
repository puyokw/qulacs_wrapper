# -*- coding: utf-8 -*-
from qulacs import QuantumState, ParametricQuantumCircuit, Observable
from gradients import get_gradient
import torch

EPS_rel = 1e-12
EPS_abs = 1e-12

class PQCfunction(torch.autograd.Function):
    """
        This function is used for implementing PQClayer.
    """
    # input size 
    # backward の返却値のsize の一致が必要
    # input is parameters of the parametrized quantum circuit
    @staticmethod
    def forward(ctx, input, data, pqc_f, obs_list, coeff_tensor, flag_data_quantum=False):
        # ctx.set_materialize_grads(False)
        n_features = data.shape[-1]
        # print(data.shape)
        ret = []
        for data_i in data:
            pqc = pqc_f(data_i, input)
            n_qubits = pqc.get_qubit_count()
            state = QuantumState(n_qubits)
            if flag_data_quantum==True:
                QuantumState.load(state, data_i.tolist()+[0]*(2**n_qubits-n_features))
            n_gates = pqc.get_gate_count()
            for i in range(n_gates):
                gate_i = pqc.get_gate(i)
                gate_i.update_quantum_state(state)
            exp_val_list = [ obs.get_expectation_value(state) for obs in obs_list]
            ret.append(exp_val_list)
        # pqc_fとobs_list はctx.save_for_backwardできない
        # 代わりに、forward のステップでgradient を求めて、
        # torch.tensor にしてsave する
        
        # data_quantum のときに、grad_inputが違う
        grad_input = []
        for data_i in data:
            pqc = pqc_f(data_i, input)
            if flag_data_quantum==True:
                n_qubits = pqc.get_qubit_count()
                state = QuantumState(n_qubits)
                QuantumState.load(state, data_i.tolist()+[0]*(2**n_qubits-n_features))
                grad_input_i = get_gradient(pqc, obs_list, state)
            else:
                grad_input_i = get_gradient(pqc, obs_list)
            # grad_input_i = ( torch.tensor(grad_input_i).T / torch.tensor(coeff_tensor) ).T
            grad_input.append( grad_input_i )
        grad_input = torch.tensor(grad_input, dtype=torch.float64)
        if len(coeff_tensor)>0:
            grad_input = grad_input/coeff_tensor.reshape(1,-1,1)
        grad_input[[grad_input.abs()<=EPS_abs]]=0
        ctx.save_for_backward(grad_input)
        ret = torch.tensor(ret, dtype=torch.float64)
        ret[[ret.abs()<=EPS_abs]]=0
        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input, = ctx.saved_tensors
        # grad_input: ( # of data, # of params, # of observable)
        # grad_output: (# of data, # of observable)
        
        grad_input_chainrule = torch.zeros(grad_input.shape[1], dtype=torch.float64)
        for grad_input_i, grad_output_i in zip(grad_input, grad_output):
            # chain rule
            # grad_input_i[[grad_input_i.abs()<=1e-16]]=0
            grad_output_i[[grad_output_i.abs()<=EPS_abs]]=0
            tmp = grad_input_i*grad_output_i
            tmp = tmp.sum(axis=1)
            tmp[[tmp.abs()<=EPS_abs]]=0
            grad_input_chainrule += tmp
            grad_input_chainrule[[grad_input_chainrule.abs()<=EPS_abs]]=0
            # grad_input_chainrule.append( tmp )
        #if len(grad_input_chainrule)!=0:
        #    ret = torch.vstack( grad_input_chainrule )
        #else:
        #    ret=None
        return grad_input_chainrule, None, None, None, None, None

# parameter 付き量子回路のparameter は
# 内部でtorch.tensor を使わないと
# かなり制限される
class PQClayer(torch.nn.Module):
    '''
        parametrized quantum circuit layer implemented by pytorch
        params: torch.tensor of parameters ( it needs required_grad=True)
        pqc: paramtrized quantum circuit
    '''
    def __init__(self, params, pqc_f, obs_list, use_parameters=True, data_quantum=False):
        super(PQClayer, self).__init__()
        self.pqc_f = pqc_f
        self.obs_list = obs_list
        self.flag_data_quantum = data_quantum
        if use_parameters==True:
            self.params = torch.nn.Parameter(params) # Optimizer で、model.parameters() のみ可
        else:
            self.params = params # Optimizer で、List でパラメータを直接指定のみ可

    def forward(self, input):
        '''
            input: tensor.torch of the data
        '''
        if len(input.shape)==1:
            pqc = self.pqc_f(input, self.params)
        elif len(input.shape)>=2:
            pqc = self.pqc_f(input[0], self.params)
        n_parametrized_gate = pqc.get_parameter_count()
        coeff_list = [ pqc.get_parameter(i)/self.params[i] for i in range(n_parametrized_gate)]
        coeff_list = torch.tensor(coeff_list)
        # params_list = [ pqc.get_parameter(i) for i in range(n_parametrized_gate)]
        # self.params = torch.nn.Parameter(torch.tensor(params_list))
        '''
        if self.flag_data_quantum==True:
            n_features = input.shape[-1]
            # print(data.shape)
            ret = []
            for data_i in input:
                pqc = self.pqc_f(data_i, self.params)
                n_qubits = pqc.get_qubit_count()
                state = QuantumState(n_qubits)
                QuantumState.load(state, data_i.tolist()+[0]*(2**n_qubits-n_features))
                n_gates = pqc.get_gate_count()
                for i in range(n_gates):
                    gate_i = pqc.get_gate(i)
                    gate_i.update_quantum_state(state)
                exp_val_list = [ obs.get_expectation_value(state) for obs in self.obs_list]
                ret.append(exp_val_list)
            ret = torch.tensor(ret)
        else:
            ret = PQCfunction.apply(self.params, input, self.pqc_f, self.obs_list, coeff_list, self.flag_data_quantum)
        '''
        return PQCfunction.apply(self.params, input, self.pqc_f, self.obs_list, coeff_list, self.flag_data_quantum)


# 1. model.parameters も使えるようにする
# 2. parameter 部分がtorch.tensorの積も許す

if __name__=='__main__':
    torch.random.manual_seed(2)
    n_output = 1
    n_qubits = 3
    state = QuantumState(n_qubits)
    def parametrized_quantum_cirucit(input, params):
        qc = ParametricQuantumCircuit(n_qubits)
        for i in range(len(input)):
            qc.add_RY_gate(i, input[i])
        qc.add_X_gate(1)
        qc.add_parametric_CRX_gate(0, 2, -params[0])
        qc.add_parametric_CRY_gate(1, 0, -params[1])
        qc.add_parametric_CRZ_gate(2, 1, -params[2])
        qc.add_CNOT_gate(0,1)
        qc.add_CNOT_gate(1,2)
        qc.add_parametric_RX_gate(2, -params[3])
        qc.add_parametric_RY_gate(1, -params[4])
        qc.add_parametric_RZ_gate(0, -params[5])
        return qc
    observable_list = []
    for i in range(min(n_output, n_qubits)):
        observable = Observable(n_qubits)
        observable.add_operator( 1.0, "Z "+str(i) )
        observable_list.append(observable)
    # make dataset
    import functools
    N = 20
    x = torch.randn(N, n_qubits, requires_grad=True)
    params = torch.randn(6, requires_grad=True)
    # pqc_layer = PQClayer(functools.partial(parametrized_quantum_cirucit, params=params), observable_list)
    pqc_layer = PQClayer(params, parametrized_quantum_cirucit, observable_list)
    model = torch.nn.Sequential(pqc_layer)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.SGD([params], lr=0.01, momentum=0.9) # この形式には対応していない
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    y = torch.randn( (N, len(observable_list)) )
    y += y.min().abs() + 1e-5
    y /= y.sum()
    for t in range(5):
        optimizer.zero_grad()
        # pqcf = PQCfunction()
        # y_pred = pqcf.apply(params, x, parametrized_quantum_cirucit, observable_list)
        y_pred  = model(x)
        # y_pred = pqc_layer.forward(x)
        # loss = torch.nn.MSELoss()
        loss = torch.nn.KLDivLoss(reduction='sum')
        y_pred += 1
        y_pred /= y_pred.sum()
        output = loss(y.log(), y_pred)
        output.backward()
        optimizer.step()
        # Compute and print loss
        # loss = (y_pred - y).pow(2).sum()
        print("step: ", t, ", loss: ", output.item())
        