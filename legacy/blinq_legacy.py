import cirq
import numpy as np
import qsimcirq
import time

from cirq.contrib.qasm_import import circuit_from_qasm
import sys
from parse import compile
from optparse import OptionParser

#### usage format
######## algo_name, qubit_size, api_option, MAX_FUSED_QSIM, MAX_FUSED_BLINQ, qsim or blinq
######## python3 blinq_legacy.py bv 5 custatevec 2 4 qsim

parser = OptionParser()
parser.add_option("-g", "--use_gpu", dest="use_gpu", default=True, help="True for GPU usage")
parser.add_option("-b", "--bench", dest="bench", help="set benchmark")
parser.add_option("-c", "--cuda_implementation", dest="CHOOSE_API", help="set cuda kernel to cuda or custatevec")
# parser.add_option("-q", "--MAX_FUSED_QSIM", dest="MAX_FUSED_QSIM", help="qsim_max_fused number")
parser.add_option("-f", "--MAX_FUSED_BLINQ", dest="MAX_FUSED_BLINQ", help="blinq_max_fused number")
# parser.add_option("-e", "--ENABLE_QSIM", dest="ENABLE_QSIM", help="True if qsim in enabled")

(options, args) = parser.parse_args()

print(options)
print(args)
### file size
# bv: 10 32 2
# adder 10 31 3
# grover: 11 31 2
# pe: 10 30 2
# qft: 10 32 2

## CHOOSE_API variable indicates API to use
## 0 for cuda, 1 for custatevec
# CHOOSE_API = 1
# CHOOSE_API = int(sys.argv[3])
if sys.argv[3]=="cuda":
    CHOOSE_API = 0
elif sys.argv[3]=="custatevec":
    CHOOSE_API = 1
else:
    raise Exception("wrong parameter in api argument !!")

## MAX_FUSED_QSIM variable indicates maximum clustering degree for QSIM
# MAX_FUSED_QSIM = 1
MAX_FUSED_QSIM = int(sys.argv[4])

## MAX_FUSED_BLINQ variable indicates maximum clustering degree for BLINQ
# MAX_FUSED_BLINQ = 4
MAX_FUSED_BLINQ = int(sys.argv[5])

## ENABLE_QSIM variable indicates qsim is run. If false, only blinq runs
if sys.argv[6]=="qsim":
    ENABLE_QSIM = True
else:
    ENABLE_QSIM = False



bench_dir="/home/ahnsungwoo/blinq/blinq_bench"
# possible bench types: adder, bv, pe, qft, grover, vqe, knn
# bench_type="grover"
bench_type=sys.argv[1]

# possible bench qnum: (adder, bv, pe, qft, grover): all odd number, knn: 31, vqe: 28
# bench_qnum="5"
bench_qnum=sys.argv[2]

bench_postfix=".qasm"
file_path=bench_dir+"/"+bench_type+"/"+bench_type+"_"+bench_qnum+bench_postfix
bench = open(file_path,mode='r')
qasm_input = bench.read()
bench.close()


#perf_prefix="/mnt/perf"
# perf_path=perf_prefix+"/"+bench_type+"_"+bench_qnum

# open(perf_path, "w").close()
# perf_file = open(perf_path,mode='a')



# perf_path2=perf_prefix+"/"+bench_type+"_"+bench_qnum+".csv"
#perf_path2=perf_prefix+"/"+"result.csv"
# open(perf_path2, "w").close()
#perf_file2 = open(perf_path2,'a', newline='')
#csv_file = csv.writer(perf_file2)

#result_bench_name=bench_type+"_"+bench_qnum

# csv_file.writerow([1, 2, 3])
# csv_file.writerow([2, 22, 33])
 

def check_size(circuit):
    checksum=True
    # print(circuit)
    for moment in circuit:
        for op in moment:
            size = pow(2, len(op.qubits))
            # print(op)
            if len(cirq.unitary(op)) != size:
                print("error")
                print(op)
                print(size)
                print(cirq.unitary(op))
                # print(cirq.unitary(cirq.CNOT))
                # print((cirq.unitary(cirq.CNOT).__class__))
                return False
                # checksum=False
    return checksum




class sw_Gate(cirq.Gate):
    def __init__(self, sw_qubits, sw_unitary, sw_str):
        self._qubits=sw_qubits
        self._unitary=sw_unitary
        self._str=sw_str
    def _num_qubits_(self) -> int:
        return self._qubits

    def _unitary_(self):
        return self._unitary

    def __str__(self):
        return self._str
    

def validate_circuit(circuit1, circuit2):
    circuit1_unitary = cirq.unitary(circuit1)
    circuit2_unitary = cirq.unitary(circuit2)
    validate = np.allclose(circuit1_unitary, circuit2_unitary)
    # print(circuit1_unitary)
    # print(circuit2_unitary)

    return validate

def validate_moment(moment1, moment2):
    moment1_unitary = cirq.unitary(moment1)
    moment2_unitary = cirq.unitary(moment2)
    validate = np.allclose(moment1_unitary, moment2_unitary)

    return validate


def test_performance_parallel(circuit, print_cirq, print_qsim, cirq_original2, qsim_original2, current_cluster_degree):
    if print_cirq:
        cirq_simulator = cirq.Simulator()
        cirq_start = time.time()
        cirq_results = cirq_simulator.simulate(circuit)
        cirq_elapsed = time.time() - cirq_start
        cirq_start = time.time()
        cirq_results = cirq_simulator.simulate(circuit)
        cirq_elapsed2 = time.time() - cirq_start
        cirq_serial_improved_perf = (cirq_original2 - cirq_elapsed2) / cirq_original2 * 100
        print(f'Cirq parallel CD {current_cluster_degree} circuit upgrade: {cirq_serial_improved_perf} %.')
        
    if print_qsim:
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed = time.time() - qsim_start
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed2 = time.time() - qsim_start
        qsim_serial_improved_perf = (qsim_original2 - qsim_elapsed2) / qsim_original2 * 100
        print(f'qsim parallel CD {current_cluster_degree} circuit upgrade: {qsim_serial_improved_perf} %.')


def test_performance_serial(circuit, print_cirq, print_qsim, cirq_original2, qsim_original2, current_cluster_degree):
    if print_cirq:
        cirq_simulator = cirq.Simulator()
        cirq_start = time.time()
        cirq_results = cirq_simulator.simulate(circuit)
        cirq_elapsed = time.time() - cirq_start
        cirq_start = time.time()
        cirq_results = cirq_simulator.simulate(circuit)
        cirq_elapsed2 = time.time() - cirq_start
        cirq_serial_improved_perf = (cirq_original2 - cirq_elapsed2) / cirq_original2 * 100
        print(f'Cirq serial   CD {current_cluster_degree} circuit upgrade: {cirq_serial_improved_perf} %.')
        
    if print_qsim:
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed = time.time() - qsim_start
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed2 = time.time() - qsim_start
        qsim_serial_improved_perf = (qsim_original2 - qsim_elapsed2) / qsim_original2 * 100
        print(f'qsim serial   CD {current_cluster_degree} circuit upgrade: {qsim_serial_improved_perf} %.')

def test_performance_insert(circuit, print_cirq, print_qsim, cirq_original2, qsim_original2, current_cluster_degree):
    if print_cirq:
        cirq_simulator = cirq.Simulator()
        cirq_start = time.time()
        cirq_results = cirq_simulator.simulate(circuit)
        cirq_elapsed = time.time() - cirq_start
        cirq_start = time.time()
        cirq_results = cirq_simulator.simulate(circuit)
        cirq_elapsed2 = time.time() - cirq_start
        cirq_serial_improved_perf = (cirq_original2 - cirq_elapsed2) / cirq_original2 * 100
        print(f'Cirq insert   CD {current_cluster_degree} circuit upgrade: {cirq_serial_improved_perf} %.')
        
    if print_qsim:
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed = time.time() - qsim_start
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed2 = time.time() - qsim_start
        qsim_serial_improved_perf = (qsim_original2 - qsim_elapsed2) / qsim_original2 * 100
        print(f'qsim insert   CD {current_cluster_degree} circuit upgrade: {qsim_serial_improved_perf} %.')
        
def test_performance_original_cirq(circuit):
    # cirq_simulator = cirq.DensityMatrixSimulator()
    cirq_simulator = cirq.Simulator()
    cirq_start = time.time()
    cirq_results = cirq_simulator.simulate(circuit)
    cirq_elapsed = time.time() - cirq_start
    cirq_start = time.time()
    cirq_results = cirq_simulator.simulate(circuit)
    cirq_elapsed2 = time.time() - cirq_start
    return cirq_elapsed2
        
def test_performance_original_qsim(circuit):
    iter=0
    qsim_start = time.time()
    qsim_results = qsim_simulator.simulate(circuit)
    qsim_elapsed = time.time() - qsim_start
    qsim_elapsed=0
    # quit()
    # print("\ncheck now\n\n")
    for i in range(iter):
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed += time.time() - qsim_start
    qsim_elapsed = qsim_elapsed/5
    # quit()
    # qsim_start = time.time()
    # qsim_results = qsim_simulator.simulate(circuit)
    # qsim_elapsed2 = time.time() - qsim_start
    # return qsim_elapsed2
    return qsim_elapsed

def test_performance_circuit_cirq(circuit):
    cirq_simulator = cirq.Simulator()
    cirq_start = time.time()
    cirq_results = cirq_simulator.simulate(circuit)
    cirq_elapsed = time.time() - cirq_start
    cirq_start = time.time()
    cirq_results = cirq_simulator.simulate(circuit)
    cirq_elapsed2 = time.time() - cirq_start
    return cirq_elapsed2

def test_performance_circuit_qsim(circuit):
    iter=0
    qsim_start = time.time()
    qsim_results = qsim_simulator.simulate(circuit)
    qsim_elapsed = time.time() - qsim_start
    qsim_elapsed=0
    for i in range(iter):
        qsim_start = time.time()
        qsim_results = qsim_simulator.simulate(circuit)
        qsim_elapsed += time.time() - qsim_start
    qsim_elapsed = qsim_elapsed/5
    # quit()
    # qsim_start = time.time()
    # qsim_results = qsim_simulator.simulate(circuit)
    # qsim_elapsed2 = time.time() - qsim_start
    # return qsim_elapsed2
    return qsim_elapsed
        

@cirq.transformer
class SerialMergeFunc:
    def __init__(self, degree):
        self._degree = degree
        
    def find_second_candidate(self, circuit, first_op, first_moment, second_list):
        second_candidate_moment = circuit.next_moment_operating_on(first_op.qubits, first_moment+1)
        if second_candidate_moment == None:
            return second_list
        else:
            second_candidate_op = circuit.operation_at(first_op.qubits[0], second_candidate_moment)
            if second_candidate_op == None:
                return second_list
            elif second_candidate_op.qubits==first_op.qubits:
                second_list.append(second_candidate_op)
                second_list = self.find_second_candidate(circuit, second_candidate_op, second_candidate_moment, second_list)
                return second_list
            
            else:
                return second_list
                
    def check_merged(self, circuit, target_op, target_moment):
        merge_status = False
        previous_moment_list = []
        previous_op_list = []
        checking_length = len(target_op.qubits)
        for index in range(0, checking_length):
            checking_qubit = [target_op.qubits[index]]
            prev_moment_index = circuit.prev_moment_operating_on(checking_qubit, target_moment)
            if prev_moment_index == None:
                return False
            else:
                previous_moment_list.append(prev_moment_index)

        check_prev_moment_value = previous_moment_list[0]
        for index in range(0, len(previous_moment_list)):
            if previous_moment_list[index] != check_prev_moment_value:
                return False
            
            previous_op = circuit.operation_at(target_op.qubits[index], previous_moment_list[index])
            previous_op_list.append(previous_op)
            
        for index in range(0, len(previous_op_list)):
            if previous_op_list[index].qubits != target_op.qubits:
                return False
                
        merge_status = True    
        return True
            

            

    def __call__(self, circuit, *, context=None):
        merged_circuit = cirq.Circuit()
        all_op = circuit.findall_operations(lambda op:op)
        for i, op in all_op:
            first_candidiate_op = op
            if len(first_candidiate_op.qubits)==self._degree:
                if self.check_merged(circuit, first_candidiate_op, i) == True:
                    continue
                second_list = []
                second_list = self.find_second_candidate(circuit, first_candidiate_op, i, second_list)
                if len(second_list) !=0:
                    new_unitary = cirq.unitary(first_candidiate_op)
                    # new_unitary = cirq.kraus(first_candidiate_op)
                    for op in second_list:
                        new_unitary = np.matmul(cirq.unitary(op), new_unitary)
                        # new_unitary = np.matmul(cirq.kraus(op), new_unitary)
                    new_gate = sw_Gate(self._degree, new_unitary, 'sm')
                    new_qubit = [qubits for qubits in first_candidiate_op.qubits]
                    merged_circuit.append(new_gate(*new_qubit))
                    
                else:
                    merged_circuit.append(first_candidiate_op)
                
            else:
                merged_circuit.append(first_candidiate_op)
        
        # print(merged_circuit)
        return merged_circuit

@cirq.transformer
class SerialMergeFunc_moment:
    def __init__(self, degree):
        self._degree = degree
        
    def find_second_candidate(self, circuit, first_op, first_moment, second_list):
        second_candidate_moment = circuit.next_moment_operating_on(first_op.qubits, first_moment+1)
        if second_candidate_moment == None:
            return second_list
        else:
            second_candidate_op = circuit.operation_at(first_op.qubits[0], second_candidate_moment)
            if second_candidate_op == None:
                return second_list
            elif second_candidate_op.qubits==first_op.qubits:
                second_list.append(second_candidate_op)
                second_list = self.find_second_candidate(circuit, second_candidate_op, second_candidate_moment, second_list)
                return second_list
            
            else:
                return second_list
                
    def check_merged(self, circuit, target_op, target_moment):
        merge_status = False
        previous_moment_list = []
        previous_op_list = []
        checking_length = len(target_op.qubits)
        for index in range(0, checking_length):
            checking_qubit = [target_op.qubits[index]]
            prev_moment_index = circuit.prev_moment_operating_on(checking_qubit, target_moment)
            if prev_moment_index == None:
                return False
            else:
                previous_moment_list.append(prev_moment_index)

        check_prev_moment_value = previous_moment_list[0]
        for index in range(0, len(previous_moment_list)):
            if previous_moment_list[index] != check_prev_moment_value:
                return False
            
            previous_op = circuit.operation_at(target_op.qubits[index], previous_moment_list[index])
            previous_op_list.append(previous_op)
            
        for index in range(0, len(previous_op_list)):
            if previous_op_list[index].qubits != target_op.qubits:
                return False
                
        merge_status = True    
        return True
            

    def __call__(self, circuit, *, context=None):
        merged_circuit = cirq.Circuit()
        # all_op = circuit.findall_operations(lambda op:op)
        
        moment_index = -1
        for moment in circuit:
            moment_index += 1
            append_moment=[]
            for op in moment:
                first_candidiate_op = op
                if len(first_candidiate_op.qubits)==self._degree:
                    if self.check_merged(circuit, first_candidiate_op, moment_index) == True:
                        continue
                    second_list = []
                    second_list = self.find_second_candidate(circuit, first_candidiate_op, moment_index, second_list)
                    if len(second_list) !=0:
                        new_unitary = cirq.unitary(first_candidiate_op)
                        # new_unitary = cirq.kraus(first_candidiate_op)
                        for op in second_list:
                            new_unitary = np.matmul(cirq.unitary(op), new_unitary)
                            # new_unitary = np.matmul(cirq.kraus(op), new_unitary)
                        new_gate = sw_Gate(self._degree, new_unitary, 'sm')
                        new_qubit = [qubits for qubits in first_candidiate_op.qubits]
                        append_moment.append(new_gate(*new_qubit))
                    else:
                        append_moment.append(first_candidiate_op)
                        
                else:
                    append_moment.append(first_candidiate_op)
                    # merged_circuit.append(first_candidiate_op)
                    
            merged_circuit.append(append_moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            # print(merged_circuit)
                
        
        # for i, op in all_op:
        #     first_candidiate_op = op
        #     if len(first_candidiate_op.qubits)==self._degree:
        #         if self.check_merged(circuit, first_candidiate_op, i) == True:
        #             continue
        #         second_list = []
        #         second_list = self.find_second_candidate(circuit, first_candidiate_op, i, second_list)
        #         if len(second_list) !=0:
        #             new_unitary = cirq.unitary(first_candidiate_op)
        #             # new_unitary = cirq.kraus(first_candidiate_op)
        #             for op in second_list:
        #                 new_unitary = np.matmul(cirq.unitary(op), new_unitary)
        #                 # new_unitary = np.matmul(cirq.kraus(op), new_unitary)
        #             new_gate = sw_Gate(self._degree, new_unitary, 'sm')
        #             new_qubit = [qubits for qubits in first_candidiate_op.qubits]
        #             merged_circuit.append(new_gate(*new_qubit))
                    
        #         else:
        #             merged_circuit.append(first_candidiate_op)
                
        #     else:
        #         merged_circuit.append(first_candidiate_op)
        
        # print(merged_circuit)
        return merged_circuit


@cirq.transformer
class SerialMergeFunc_collapse:
    def __init__(self, degree):
        self._degree = degree
        
    def construct_collapsed_circuit_left(self, circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op):  
        collapsed_circuit = cirq.Circuit()
        
        for moment in circuit[:second_candidate_moment_index]:
            collapsed_circuit.append(moment)
            
        target_moment = circuit[second_candidate_moment_index]
        replaced_moment = target_moment.without_operations_touching(second_candidate_op.qubits)
        collapsed_circuit.append(replaced_moment)
            
        for moment in circuit[second_candidate_moment_index+1:]:
            collapsed_circuit.append(moment)
        
        # print(first_op)
        # print(second_candidate_op)
        # print(collapsed_circuit)
        return collapsed_circuit
    
    def check_collapse_possible_left(self, circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op):
        for second_index in range(len(second_candidate_op.qubits)):
            check_op_index = circuit.next_moment_operating_on([second_candidate_op.qubits[second_index]], second_candidate_moment_index+1)
            if check_op_index != first_moment_index:
                return False
            if first_op != circuit.operation_at(second_candidate_op.qubits[second_index], check_op_index):
                return False
            
        return True
    
    def collapse_left(self, circuit, first_moment_index, first_op):
        
        for index in range(len(first_op.qubits)):
            second_candidate_moment_index = circuit.prev_moment_operating_on([first_op.qubits[index]], first_moment_index)
            if second_candidate_moment_index != None:
                second_candidate_op = circuit.operation_at(first_op.qubits[index], second_candidate_moment_index)
                collapse_possible = self.check_collapse_possible_left(circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op)
                
                if collapse_possible:
                    collapsed_circuit = self.construct_collapsed_circuit_left(circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op)
                    return collapsed_circuit, True
        
        return circuit, False


    def construct_collapsed_circuit_right(self, circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op):  
        collapsed_circuit = cirq.Circuit()
        
        for moment in circuit[:second_candidate_moment_index]:
            collapsed_circuit.append(moment)
            
        target_moment = circuit[second_candidate_moment_index]
        replaced_moment = target_moment.without_operations_touching(second_candidate_op.qubits)
        collapsed_circuit.append(replaced_moment)
            
            
        if len(circuit) > second_candidate_moment_index+1:
            for moment in circuit[second_candidate_moment_index+1:]:
                collapsed_circuit.append(moment)
        
        return collapsed_circuit
    def check_collapse_possible_right(self, circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op):
        
        for second_index in range(len(second_candidate_op.qubits)):
            check_op_index = circuit.prev_moment_operating_on([second_candidate_op.qubits[second_index]], second_candidate_moment_index)
            if check_op_index != first_moment_index:
                return False
            if first_op != circuit.operation_at(second_candidate_op.qubits[second_index], check_op_index):
                return False
            
        return True
    def collapse_right(self, circuit, first_moment_index, first_op):
        
        for index in range(len(first_op.qubits)):
            second_candidate_moment_index = circuit.next_moment_operating_on([first_op.qubits[index]], first_moment_index+1)
            if second_candidate_moment_index != None:
                second_candidate_op = circuit.operation_at(first_op.qubits[index], second_candidate_moment_index)
                collapse_possible = self.check_collapse_possible_right(circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op)
                
                if collapse_possible:
                    collapsed_circuit = self.construct_collapsed_circuit_right(circuit, first_moment_index, first_op, second_candidate_moment_index, second_candidate_op)
                    return collapsed_circuit, True
        
        return circuit, False
        
            
    def merge_collapse(self, circuit):
        
        changed = False
        
        all_op = circuit.findall_operations(lambda op:op)
        for first_moment_index, first_op in all_op:
            circuit, changed = self.collapse_left(circuit, first_moment_index, first_op)
            if changed:
                # print(circuit)
                return circuit, True
            
            circuit, changed = self.collapse_right(circuit, first_moment_index, first_op)
            if changed:
                # print(circuit)
                return circuit, True
        
        return circuit, False
        
    def __call__(self, circuit, *, context=None):
        collapsed_circuit=cirq.Circuit()
        changed = True
        while changed == True:
            circuit, changed = self.merge_collapse(circuit)
        
        for moment in circuit:
            if len(moment)!=0:
                collapsed_circuit.append(moment)
            
        # merged_circuit = circuit
        return collapsed_circuit


@cirq.transformer
class ParallelMergeFunc_full:
    def __init__(self, degree):
        self._degree = degree
            

    
    def get_coefficient(self, op1, op2, op1_qubit_index, op2_qubit_index, c_row, c_col):
        
        # required qubit index list
        # op_index_original
        # op_index_shrinked_inter
        # op_index_shrinked_intra
        
        op1_index_original = op1_qubit_index
        op2_index_original = op2_qubit_index
        
        # if (op2_qubit_index==[2, 3, 4]):
        #     print(op2)
        
        op1_index_shrinked_inter = []
        op2_index_shrinked_inter = []
        total_index = sorted(op1_index_original + op2_index_original)
        op1_len = len(op1_qubit_index)
        op2_len = len(op2_qubit_index)
        
        for index in range(len(total_index)):
            if total_index[index] in op1_index_original:
                op1_index_shrinked_inter.append(index)
            else:
                op2_index_shrinked_inter.append(index)
        
        op1_index_shrinked_intra = []
        op2_index_shrinked_intra = []
        
        for index in range(op1_len):
            op1_index_shrinked_intra.append(index)
        for index in range(op2_len):
            op2_index_shrinked_intra.append(index)
                
        # if (op2_qubit_index==[2, 3, 4]):
        #     print(op2_index_shrinked_inter)        
        #     print(op2_index_shrinked_intra) 
        #     print()       
                
        # c_row represents the actual final state itself
        # the result of this "get_coefficient" represents the value from "c_col" state to "c_row"
        # transition represents c_col to c_row
        c_col_bin = [int(x) for x in bin(c_col)[2:]]
        c_row_bin = [int(x) for x in bin(c_row)[2:]]
        
        while (len(c_col_bin) != len(total_index)):
            c_col_bin.insert(0, 0)
        while (len(c_row_bin) != len(total_index)):
            c_row_bin.insert(0, 0)
        
        op1_row_bin = []
        op1_col_bin = []
        
        for index in op1_index_shrinked_inter:
            op1_row_bin.append(c_row_bin[index])
            op1_col_bin.append(c_col_bin[index])
            
        op2_row_bin = []
        op2_col_bin = []
        
        
        for index in op2_index_shrinked_inter:
            op2_row_bin.append(c_row_bin[index])
            op2_col_bin.append(c_col_bin[index])
        
        
        op1_row_coefficient = 0
        op1_col_coefficient = 0
        
        for index in range(op1_len):
            op1_row_coefficient += op1_row_bin[index] << op1_index_shrinked_intra[op1_len-index-1]
            op1_col_coefficient += op1_col_bin[index] << op1_index_shrinked_intra[op1_len-index-1]
            
        op2_row_coefficient = 0
        op2_col_coefficient = 0
        
        for index in range(op2_len):
            op2_row_coefficient += op2_row_bin[index] << op2_index_shrinked_intra[op2_len-index-1] 
            op2_col_coefficient += op2_col_bin[index] << op2_index_shrinked_intra[op2_len-index-1] 
        
        op1_coefficient = cirq.unitary(op1)[op1_row_coefficient][op1_col_coefficient]
        op2_coefficient = cirq.unitary(op2)[op2_row_coefficient][op2_col_coefficient]
        
        
        final_coefficient = op1_coefficient * op2_coefficient
        
        # if c_row==5 and c_col==7:
        #     if (op2_qubit_index==[2, 3, 4]):
        #         # print(final_coefficient)
        #         print("index: ", c_row, c_col)
        #         print(op1_row_coefficient)
        #         print(op1_col_coefficient)
        #         print(op2_row_coefficient)
        #         print(op2_col_coefficient)
        #         print(op2_coefficient)
        #         # print(cirq.unitary(op2))
                
                # print()
        
        return final_coefficient
    
    def make_clustered_op(self, op1, op2):
        
        
        op1_qubits = [qubits for qubits in op1.qubits]
        op2_qubits = [qubits for qubits in op2.qubits]
        
        new_op_qubit = sorted(op1_qubits + op2_qubits)
        new_op_unitary_dim = pow(2, len(new_op_qubit))
        
        parse_qubit_index = compile("q_{}")
        # parse_qubit_index = compile("q{}")
        
        op1_qubit_index=[]
        op2_qubit_index=[]
        
        for index in op1_qubits:
            op1_qubit_index.append(int((parse_qubit_index.parse(str(index)))[0]))
        for index in op2_qubits:
            op2_qubit_index.append(int((parse_qubit_index.parse(str(index)))[0]))
        
        # if (op2_qubit_index==[2, 4, 3]):
        #     print(cirq.unitary(op2))
        #     print("euraka")
        # op1_qubit_index=sorted(op1_qubit_index)
        # op2_qubit_index=sorted(op2_qubit_index)
        
        # if (op2_qubit_index==[2, 3, 4]):
        #     print("sorted")
        
        new_op_unitary_row = []
        new_op_unitary = np.empty((0,new_op_unitary_dim), float)   
        
        for c_row in range(new_op_unitary_dim):
            new_op_unitary_row = []
            for c_col in range(new_op_unitary_dim):
                
                final_coefficient = self.get_coefficient(op1, op2, op1_qubit_index, op2_qubit_index, c_row, c_col)
                
                new_op_unitary_row.append(final_coefficient)
            new_op_unitary = np.append(new_op_unitary, np.array([new_op_unitary_row]), axis=0)

        new_gate = sw_Gate(self._degree, new_op_unitary, 'pm')
        new_op = new_gate(*new_op_qubit)
        return new_op
    
        

    def __call__(self, circuit, context=None):
        merged_circuit = cirq.Circuit()
        qubit_tuple = sorted(circuit.all_qubits())
        for moment in circuit:
            new_moment = []
            merged_op_list = []
            non_affecting_op_list=[]
            moment_op_list = []
            for qubit in qubit_tuple:
                for op in moment:
                    if qubit in op.qubits and op not in moment_op_list:
                        moment_op_list.append(op)
            cluster_candidate_op_list = moment_op_list
            
            if(len(cluster_candidate_op_list)>=2):
                for op1_index in range(len(cluster_candidate_op_list)-1):
                    op1_candidate = cluster_candidate_op_list[op1_index]
                    
                    # if op1_candidate in non_affecting_op_list:
                    #     continue
                    
                    for op2_index in range(op1_index+1, len(cluster_candidate_op_list)):
                        op2_candidate = cluster_candidate_op_list[op2_index]
                        
                        # if op2_candidate in non_affecting_op_list:
                        #     continue
                        
                        if op1_candidate in merged_op_list:
                            if (op1_index == len(cluster_candidate_op_list)-2 and op2_index == len(cluster_candidate_op_list)-1):
                                new_moment.append(op2_candidate)
                            continue
                        if op2_candidate in merged_op_list:
                            continue
                        if len(op1_candidate.qubits) + len(op2_candidate.qubits) != self._degree:
                            if (op1_index == len(cluster_candidate_op_list)-2 and op2_index == len(cluster_candidate_op_list)-1):
                                new_moment.append(op2_candidate)
                            continue
                        
                        
                        
                        
                        op1_qubits = [qubits for qubits in op1_candidate.qubits]
                        op2_qubits = [qubits for qubits in op2_candidate.qubits]

                        parse_qubit_index = compile("q_{}")
                        # parse_qubit_index = compile("q{}")

                        op1_qubit_index=[]
                        op2_qubit_index=[]

                        for index in op1_qubits:
                            op1_qubit_index.append(int((parse_qubit_index.parse(str(index)))[0]))
                        for index in op2_qubits:
                            op2_qubit_index.append(int((parse_qubit_index.parse(str(index)))[0]))
                            
                        if op1_qubit_index!=sorted(op1_qubit_index):
                            if op1_candidate in non_affecting_op_list:
                                pass
                            else:
                                new_moment.append(op1_candidate)
                                non_affecting_op_list.append(op1_candidate)
                                
                            if op2_qubit_index!=sorted(op2_qubit_index):
                                if op2_candidate in non_affecting_op_list:
                                    pass
                                else:
                                    new_moment.append(op2_candidate)
                                    non_affecting_op_list.append(op2_candidate)
                            elif op2_index == len(cluster_candidate_op_list)-1:
                                new_moment.append(op2_candidate)
                            else:
                                pass
                            continue
                            
                        if op2_qubit_index!=sorted(op2_qubit_index):
                            if op2_candidate in non_affecting_op_list:
                                pass
                            else:
                                new_moment.append(op2_candidate)
                                non_affecting_op_list.append(op2_candidate)
                            continue
                        
                        
                        
                        
                        merged_op_list.append(op1_candidate)
                        merged_op_list.append(op2_candidate)
                        
                        # print(op1_candidate)
                        # print(op2_candidate)
                        # print(op2_candidate.qubits)
                        # print(cirq.unitary(op2_candidate))
                        new_op=self.make_clustered_op(op1_candidate, op2_candidate)
                    
                        new_moment.append(new_op)
                    
                    if op1_candidate in non_affecting_op_list:
                        continue
                    
                    if op1_candidate not in merged_op_list:
                        new_moment.append(op1_candidate)
                        
                        if op1_index == len(cluster_candidate_op_list)-1:
                            new_moment.append(op2_candidate)
                            
            else:
                new_moment = moment
            merged_circuit.append(new_moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            # print(merged_circuit)
        return merged_circuit
   

@cirq.transformer
class ParallelMergeFunc:
    def __init__(self, degree):
        self._degree = degree
            
    def make_new_op(self, op1, op2):
        new_unitary = np.kron(cirq.unitary(op1), cirq.unitary(op2))
        # new_unitary = np.kron(cirq.kraus(op1), cirq.kraus(op2))
        new_gate = sw_Gate(self._degree, new_unitary, 'pm')
        
        new_qubit_1 = [qubits for qubits in op1.qubits]
        new_qubit_2 = [qubits for qubits in op2.qubits]
        new_qubit = new_qubit_1 + new_qubit_2
        
        new_op = new_gate(*new_qubit)
        return new_op

    def __call__(self, circuit, context=None):
        merged_circuit = cirq.Circuit()
        # print(sorted(circuit.all_qubits()))
        # qubit_tuple = tuple(circuit.all_qubits())
        qubit_tuple = sorted(circuit.all_qubits())
        # print(qubit_tuple)
        for moment in circuit:
            new_moment = []
            merged_op_list = []
            moment_op_list = []
            for qubit in qubit_tuple:
                for op in moment:
                    if qubit in op.qubits and op not in moment_op_list:
                        moment_op_list.append(op)
            cluster_candidate_op_list = moment_op_list
            # print(moment)
            # print(cluster_candidate_op_list)
            if(len(cluster_candidate_op_list)>=2):
                for index in range(0, len(cluster_candidate_op_list[:-1])):
                    op1_candidate = cluster_candidate_op_list[index]
                    op2_candidate = cluster_candidate_op_list[index+1]
                    if op1_candidate in merged_op_list:
                        if index == len(cluster_candidate_op_list)-2:
                            new_moment.append(op2_candidate)
                    else:
                        if len(op1_candidate.qubits) + len(op2_candidate.qubits) == self._degree:
                            merged_op_list.append(op2_candidate)
                            new_op = self.make_new_op(op1_candidate, op2_candidate)
                            # print(moment)
                            # print(op1_candidate)
                            new_moment.append(new_op)
                        elif index == len(cluster_candidate_op_list)-2:
                            new_moment.append(op1_candidate)
                            new_moment.append(op2_candidate)
                        else:
                            new_moment.append(op1_candidate)
            else:
                new_moment = moment
            merged_circuit.append(new_moment)
        return merged_circuit
   
@cirq.transformer
class ParallelMergeFunc_prev:
    def __init__(self, degree):
        self._degree = degree
            
    def make_new_op(self, op1, op2):
        new_unitary = np.kron(cirq.unitary(op1), cirq.unitary(op2))
        new_gate = sw_Gate(self._degree, new_unitary, 'pm')
        
        new_qubit_1 = [qubits for qubits in op1.qubits]
        new_qubit_2 = [qubits for qubits in op2.qubits]
        new_qubit = new_qubit_1 + new_qubit_2
        
        new_op = new_gate(*new_qubit)
        return new_op

    def __call__(self, circuit, *, context=None):
        merged_circuit = cirq.Circuit()
        for moment in circuit:
            new_moment = []
            merged_op_list = []
            cluster_candidate_op_list = moment.operations
            if(len(cluster_candidate_op_list)>=2):
                for index in range(0, len(cluster_candidate_op_list[:-1])):
                    op1_candidate = cluster_candidate_op_list[index]
                    op2_candidate = cluster_candidate_op_list[index+1]
                    
                    if op1_candidate in merged_op_list:
                        if index == len(cluster_candidate_op_list)-2:
                            new_moment.append(op2_candidate)
                    
                    else:
                        if len(op1_candidate.qubits) + len(op2_candidate.qubits) == self._degree:
                            merged_op_list.append(op2_candidate)
                            new_op = self.make_new_op(op1_candidate, op2_candidate)
                            new_moment.append(new_op)
                        elif index == len(cluster_candidate_op_list)-2:
                            new_moment.append(op1_candidate)
                            new_moment.append(op2_candidate)
                        else:
                            new_moment.append(op1_candidate)
                
            else:
                new_moment = moment
            
            merged_circuit.append(new_moment)
        
        # print(merged_circuit)
        return merged_circuit
   

@cirq.transformer
class InsertIdentityFunc:
    def __init__(self, degree):
        self._degree = degree
            

    def target_insertion(self, circuit, moment, moment_index, max_cluster_degree):
        insert_position = []
        inserted_moment = moment
        
        current_moment_op_qubit = []
        current_moment_noop_qubit = []
        all_position = list(circuit.all_qubits())
        current_moment = moment
        for qubit in all_position:
            if current_moment.operates_on([qubit]):
                current_moment_op_qubit.append(qubit)
            else:
                current_moment_noop_qubit.append(qubit)
        

        if moment_index == 0:
            return current_moment
        else:
            prev_moment = circuit[moment_index-1]
            for qubit in current_moment_noop_qubit:
                if prev_moment.operates_on([qubit]):
                    insert_position.append(qubit)
                else:
                    pass
            

        insert_count = len(current_moment_op_qubit)


        insert_position_set = frozenset(set(insert_position) | set(current_moment_op_qubit))

        inserted_moment = current_moment.expand_to(insert_position_set)
        
        return inserted_moment

    def __call__(self, circuit, *, context=None):
        modified_circuit = cirq.Circuit()
        
        max_cluster_degree = self._degree
        
        moment_index = -1
        for moment in circuit:
            moment_index += 1
            insert_flag = True
            for op in moment.operations:
                if len(op.qubits) >= max_cluster_degree:
                    insert_flag = False

            if insert_flag:
                inserted_moment = self.target_insertion(circuit, moment, moment_index, max_cluster_degree)
                modified_circuit.append(inserted_moment)
                     
            else:
                modified_circuit.append(moment)
        return modified_circuit    

@cirq.transformer
class InsertFrontAndBack:
    def __init__(self, degree):
        self._degree = degree
            

    def target_insertion(self, circuit, moment_index):
        current_moment=circuit[moment_index]
        
        current_moment_op_qubit = []
        current_moment_noop_qubit = []
        insert_position = []
    
        all_position = list(circuit.all_qubits())
        
        for qubit in all_position:
            if current_moment.operates_on([qubit]):
                current_moment_op_qubit.append(qubit)
            else:
                current_moment_noop_qubit.append(qubit)
        
        if(moment_index==0):
            next_moment = circuit[moment_index+1]
            for qubit1 in current_moment_noop_qubit:
                if next_moment.operates_on([qubit1]):
                    insert_position.append(qubit1)

        elif(moment_index==len(circuit)-1):
            prev_moment = circuit[moment_index-1]
            for qubit2 in current_moment_noop_qubit:
                if prev_moment.operates_on([qubit2]):
                    insert_position.append(qubit2)

        else:    
            prev_moment = circuit[moment_index-1]
            next_moment = circuit[moment_index+1]
            for qubit3 in current_moment_noop_qubit:
                if prev_moment.operates_on([qubit3]):
                    insert_position.append(qubit3)

            for qubit4 in current_moment_noop_qubit:
                if next_moment.operates_on([qubit4]):
                    insert_position.append(qubit4)


        insert_position_set = frozenset(set(insert_position) | set(current_moment_op_qubit))

        inserted_moment = current_moment.expand_to(insert_position_set)
        print()
        return inserted_moment

    def __call__(self, circuit, *, context=None):
        modified_circuit = cirq.Circuit()
        
        moment_index = -1
        for moment in circuit:
            moment_index += 1
            inserted_moment = self.target_insertion(circuit, moment_index)
            modified_circuit.append(inserted_moment)
                     
        return modified_circuit    


@cirq.transformer
class InsertFrontAndBack_without_expand:
    def __init__(self, degree):
        self._degree = degree
            

    def target_insertion(self, circuit, moment_index):
        current_moment=circuit[moment_index]
        
        current_moment_op_qubit = []
        current_moment_noop_qubit = []
        insert_position = []
    
        all_position = list(circuit.all_qubits())
        
        for qubit in all_position:
            if current_moment.operates_on([qubit]):
                current_moment_op_qubit.append(qubit)
            else:
                current_moment_noop_qubit.append(qubit)
        
        if(moment_index==0):
            if len(circuit) == 1:
                pass
            else:
                next_moment = circuit[moment_index+1]
                for qubit1 in current_moment_noop_qubit:
                    if next_moment.operates_on([qubit1]):
                        insert_position.append(qubit1)

        elif(moment_index==len(circuit)-1):
            prev_moment = circuit[moment_index-1]
            for qubit2 in current_moment_noop_qubit:
                if prev_moment.operates_on([qubit2]):
                    insert_position.append(qubit2)

        else:    
            prev_moment = circuit[moment_index-1]
            next_moment = circuit[moment_index+1]
            for qubit3 in current_moment_noop_qubit:
                if prev_moment.operates_on([qubit3]):
                    insert_position.append(qubit3)

            for qubit4 in current_moment_noop_qubit:
                if next_moment.operates_on([qubit4]):
                    insert_position.append(qubit4)

        insert_position_shrink=[]
        for index in insert_position:
            if index not in insert_position_shrink:
                insert_position_shrink.append(index)

        identity_op_unitary = np.array([[0,1], [1, 0]])   
        inserted_moment = current_moment
        
        # print("start")
        # print(insert_position_shrink)
        for index in insert_position_shrink:
            index_to_list = []
            index_to_list.append(index)
            
            identity_gate = sw_Gate(1, identity_op_unitary, 'iden')
            identity_op = identity_gate(*index_to_list)
            inserted_moment=inserted_moment.with_operation(identity_op)
        return inserted_moment


    def __call__(self, circuit, *, context=None):
        modified_circuit = cirq.Circuit()
        
        moment_index = -1
        for moment in circuit:
            moment_index += 1
            inserted_moment = self.target_insertion(circuit, moment_index)
            modified_circuit.append(inserted_moment)
                     
        return modified_circuit    

### inserts identity by watching odd and even index moments dual in only.
@cirq.transformer
class Insert_dual:
    def __init__(self, degree):
        self._degree = degree
            
    def target_insertion_odd(self, circuit, moment_index):
        current_moment=circuit[moment_index]
        
        current_moment_op_qubit = []
        current_moment_noop_qubit = []
        insert_position = []
    
        all_position = list(circuit.all_qubits())
        
        for qubit in all_position:
            if current_moment.operates_on([qubit]):
                current_moment_op_qubit.append(qubit)
            else:
                current_moment_noop_qubit.append(qubit)
        

        prev_moment = circuit[moment_index-1]
        for qubit3 in current_moment_noop_qubit:
            if prev_moment.operates_on([qubit3]):
                insert_position.append(qubit3)


        identity_op_unitary = np.array([[0,1], [1, 0]])   
        inserted_moment = current_moment
        
        if len(current_moment_op_qubit) + len(insert_position)> self._degree:
            pass
        else:
            for index in insert_position:
                index_to_list = []
                index_to_list.append(index)

                identity_gate = sw_Gate(1, identity_op_unitary, 'I')
                identity_op = identity_gate(*index_to_list)
                inserted_moment=inserted_moment.with_operation(identity_op)
                
        return inserted_moment


    def target_insertion_even(self, circuit, moment_index):
        current_moment=circuit[moment_index]
        
        current_moment_op_qubit = []
        current_moment_noop_qubit = []
        insert_position = []
    
        all_position = list(circuit.all_qubits())
        
        for qubit in all_position:
            if current_moment.operates_on([qubit]):
                current_moment_op_qubit.append(qubit)
            else:
                current_moment_noop_qubit.append(qubit)
        
        next_moment = circuit[moment_index+1]
        for qubit4 in current_moment_noop_qubit:
            if next_moment.operates_on([qubit4]):
                insert_position.append(qubit4)


        identity_op_unitary = np.array([[1,0], [0, 1]])   
        inserted_moment = current_moment
        
        if len(current_moment_op_qubit) + len(insert_position)> self._degree:
            pass
        else:
            for index in insert_position:
                index_to_list = []
                index_to_list.append(index)

                identity_gate = sw_Gate(1, identity_op_unitary, 'I')
                identity_op = identity_gate(*index_to_list)
                inserted_moment=inserted_moment.with_operation(identity_op)
                
        return inserted_moment


    def __call__(self, circuit, *, context=None):
        modified_circuit = cirq.Circuit()
        
        end_moment_index = len(circuit)
        if len(circuit)%2==1:
            end_moment_index=len(circuit)-1
            
        moment_index = -1
        for moment in circuit[:end_moment_index]:
            moment_index += 1
            if moment_index%2==0:
                # print(moment_index)
                inserted_moment = self.target_insertion_even(circuit, moment_index)
            else:
                inserted_moment = self.target_insertion_odd(circuit, moment_index)
            modified_circuit.append(inserted_moment)
                     
        if len(circuit)%2==1:
            modified_circuit.append(circuit[-1])          
                     
        return modified_circuit    


@cirq.transformer
class CheckIdentityPosition:
    def __init__(self, degree):
        # self._degree = degree
        pass
            

    def target_insertion(self, circuit, moment, moment_index, max_cluster_degree):
        insert_position = []
        

        
        return insert_position

    def __call__(self, circuit, *, context=None):
        inserted_circuit = cirq.Circuit()
        # identity_position = identity_insertion()
        # ... represented as list
        inserted_position = []
        inserted_circuit = circuit
        for qubit in inserted_circuit:
            for moment in inserted_circuit[qubit]:
                # inserted_circuit.insert()
                pass
        
        return inserted_circuit    

def conduct_clustering(circuit, max_cluster_degree):
    serial_func = SerialMergeFunc(1)
    clustered_circuit = serial_func(circuit)
    
    for degree in range(2,max_cluster_degree+1):
        parallel_func=ParallelMergeFunc_full(degree)
        serial_func=SerialMergeFunc(degree)
        
        clustered_circuit = parallel_func(clustered_circuit)
        clustered_circuit = serial_func(clustered_circuit)
        
    return clustered_circuit

def conduct_clustering_for_insert(circuit, max_cluster_degree):
    # serial_func = SerialMergeFunc(1)
    # clustered_circuit = serial_func(circuit)
    clustered_circuit = circuit
    
    for degree in range(2,max_cluster_degree+1):
        parallel_func=ParallelMergeFunc_full(degree)
        serial_func=SerialMergeFunc_moment(degree)
        
        clustered_circuit = parallel_func(clustered_circuit)
        clustered_circuit = serial_func(clustered_circuit)
        
    return clustered_circuit

def conduct_clustering_collapse(circuit, max_cluster_degree):
    serial_func = SerialMergeFunc_collapse(1)
    clustered_circuit = serial_func(circuit)
    # print(clustered_circuit)
    
    for degree in range(2,max_cluster_degree+1):
        parallel_func=ParallelMergeFunc_full(degree)
        serial_func=SerialMergeFunc_collapse(degree)
        
        clustered_circuit = parallel_func(clustered_circuit)
        # print(clustered_circuit)
        clustered_circuit = serial_func(clustered_circuit)
        # print(clustered_circuit)
        
    return clustered_circuit


def apply_BlinQ_module(circuit, max_cluster_degree):


    if ENABLE_QSIM:    
        qsim_original = test_performance_original_qsim(circuit)
        return circuit

    # print(f'qsim original      circuit runtime: {qsim_original} seconds.')
    
    max_cluster_degree = min(max_cluster_degree, len(circuit.all_qubits()))
    
    print_cirq = False
    # print_qsim = True
    print_qsim = not ENABLE_QSIM
    

    # clustered_circuit=conduct_clustering(circuit, max_cluster_degree)
    clustered_circuit=conduct_clustering_collapse(circuit, max_cluster_degree)
    # print(clustered_circuit)

    ######### insert function ##############
    identity = True
    if identity:
        iter = max_cluster_degree
        
        # insert_func = InsertFrontAndBack_without_expand(max_cluster_degree)
        
        # insert_func = Insert_dual(max_cluster_degree)
        # inserted_circuit = insert_func(clustered_circuit)
        for _ in range(iter):
            insert_func = Insert_dual(max_cluster_degree)
            inserted_circuit = insert_func(clustered_circuit)
            clustered_circuit=conduct_clustering_for_insert(inserted_circuit, max_cluster_degree)
            
            # clustered_circuit=conduct_clustering(inserted_circuit, max_cluster_degree)
            # clustered_circuit=conduct_clustering_collapse(inserted_circuit, max_cluster_degree)

            
    
    if print_cirq:
        cirq_optimized = test_performance_circuit_cirq(circuit)
        print(f'Cirq optimized      circuit runtime: {cirq_optimized} seconds.')
        cirq_improved_perf = (cirq_original - cirq_optimized) / cirq_original * 100
        print(f'Cirq  circuit upgrade: {cirq_improved_perf} %.')
    
    if print_qsim:
        qsim_optimized = test_performance_circuit_qsim(clustered_circuit)
        # print(f'qsim optimized     circuit runtime: {qsim_optimized} seconds.')
    # print()

    return clustered_circuit



def apply_BlinQ_prev(circuit, max_cluster_degree):
    
    max_cluster_degree = min(max_cluster_degree, len(circuit.all_qubits()))
    current_cluster_degree = 1
    
    print_cirq = False
    print_qsim = True
    

    # cirq_original = test_performance_original_cirq(circuit)
    qsim_original = test_performance_original_qsim(circuit)

    # print(f'Cirq original      circuit runtime: {cirq_original} seconds.')
    # print(f'qsim original      circuit runtime: {qsim_original} seconds.')
    
    
    print(f'qsim original      circuit runtime: {qsim_original} seconds.')
    # write = "qsim original circuit runtime: "+ str(qsim_original) + " seconds\n"
    # perf_file.write(write)

    # csv_file.writerow(["qsim original circuit runtime", qsim_original])
    
    serial_func = SerialMergeFunc(1)
    serial_circuit = serial_func(circuit)
    
    # print(serial_circuit)
    
    # test_performance_serial(serial_circuit, print_cirq, print_qsim, cirq_original, qsim_original, current_cluster_degree)

    
    for cluster_degree in range(2, max_cluster_degree+1):
        # parallel_func = ParallelMergeFunc(cluster_degree)
        parallel_func = ParallelMergeFunc_full(cluster_degree)
        parallel_circuit = parallel_func(serial_circuit)
        
        # print(parallel_circuit)
        
        
        serial_func = SerialMergeFunc(cluster_degree)
        serial_circuit = serial_func(parallel_circuit)
        
        # print(serial_circuit)
      

    
        
    final_circuit = serial_circuit
    # print()
    # print()
    # print(final_circuit)

    ######### insert function, but not complete ##############
    identity = False
    if identity:
        # insert_func = InsertIdentityFunc(max_cluster_degree)
        # insert_func = InsertFrontAndBack(max_cluster_degree)
        insert_func = InsertFrontAndBack_without_expand(max_cluster_degree)
        inserted_circuit = insert_func(final_circuit)
        
        # print("Validation for insert: ", validate_circuit(circuit, inserted_circuit))
        # print(final_circuit)
        # print(inserted_circuit)

        serial_func = SerialMergeFunc(1)
        serial_circuit = serial_func(inserted_circuit)
        
        # print("Validation for serial: ", validate_circuit(circuit, serial_circuit))

        # test_performance_insert(inserted_circuit, print_cirq, print_qsim, cirq_original, qsim_original, current_cluster_degree)

        for cluster_degree in range(2, max_cluster_degree+1):
            # parallel_func = ParallelMergeFunc(cluster_degree)
            parallel_func = ParallelMergeFunc_full(cluster_degree)
            parallel_circuit = parallel_func(serial_circuit)
            
            # print(parallel_circuit)
            # print("Validation for parallel: ", validate_circuit(circuit, parallel_circuit))
            # if validate_circuit(circuit, parallel_circuit) == False:
            #     print(parallel_circuit)
            #     for index in range(len(serial_circuit)):
            #         print("Validation for moment: ", validate_circuit(serial_circuit[index], parallel_circuit[index]))
                    # print(serial_circuit[index])
                    # print(parallel_circuit[index])

            # test_performance_parallel(parallel_circuit, print_cirq, print_qsim, cirq_original, qsim_original, cluster_degree)

            serial_func = SerialMergeFunc(cluster_degree)
            serial_circuit = serial_func(parallel_circuit)
            
            # print(serial_circuit)
            # print("Validation for serial: ", validate_circuit(circuit, serial_circuit))

            # test_performance_serial(serial_circuit, print_cirq, print_qsim, cirq_original, qsim_original, cluster_degree)

        final_circuit = serial_circuit
    
    if print_cirq:
        cirq_optimized = test_performance_circuit_cirq(circuit)
        print(f'Cirq optimized      circuit runtime: {cirq_optimized} seconds.')
        cirq_improved_perf = (cirq_original - cirq_optimized) / cirq_original * 100
        print(f'Cirq  circuit upgrade: {cirq_improved_perf} %.')
    
    if print_qsim:
        qsim_optimized = test_performance_circuit_qsim(final_circuit)
        print(f'qsim optimized     circuit runtime: {qsim_optimized} seconds.')
        # write = "qsim optimized circuit runtime: "+ str(qsim_optimized) + " seconds\n"
        # perf_file.write(write)
        # csv_file.writerow(["qsim optimized circuit runtime", qsim_optimized])
        
        # qsim_improved_perf = (qsim_original - qsim_optimized) / qsim_original * 100
        # print(f'qsim  circuit upgrade: {qsim_improved_perf} %.')
        # write = "qsim  circuit upgrade: "+ str(qsim_improved_perf) + " percent\n\n"
        # perf_file.write(write)
        
        # csv_file.writerow(["original", qsim_original, "optimized", qsim_optimized, "improved ratio", qsim_improved_perf])
        # csv_file.writerow([result_bench_name, qsim_original, qsim_optimized, qsim_improved_perf])
    print()

    return final_circuit


def find_sparsity(unitary, dimension):
    zero_count = 0
    for row in range(0, dimension):
        for col in range(0, dimension):
            if np.absolute(unitary[row][col]) == 0:
                zero_count += 1
                
    sparsity = zero_count / (dimension ** 2) * 100
    return sparsity
    
def get_gate_count(circuit, size):
    gate_count = 0
    for i, op in circuit.findall_operations(lambda op:len(op.qubits)==size):
        gate_count += 1
    return gate_count

def circuit_analysis(circuit, max_degree):
    single_qubit_gate_count = get_gate_count(circuit, 1)
    two_qubit_gate_count = 0
    three_qubit_gate_count = 0
    
    total_ops = list(circuit.all_operations())
    # print("total number of gates: ", len(total_ops))
    
    
    
    # search_number = len(circuit.all_qubits())
    search_number = max_degree
    search_list = []
    sum = 0
    for index in range(1, search_number+1):
        temp = get_gate_count(circuit, index)
        # print("number of ", index, " qubit gates: ", get_gate_count(circuit, index))
        sum += temp
    # print("sum gates: ", sum)
    
    # print()
    
    
    count_op = [0 for i in range(0, max(max_degree+1, 4))]
    sparsity_list = []
    all_op = circuit.findall_operations(lambda op:op)
    # print("Optimized Circuit: ")
    for i, op in all_op:
        sparsity_list.append(find_sparsity(cirq.unitary(op), pow(2, len(op.qubits))))
        # sparsity_list.append(find_sparsity(cirq.kraus(op), pow(2, len(op.qubits))))
        count_op[len(op.qubits)] += 1
    

    # print(count_op)

    # print("sparsity for each op")
    # print(sparsity_list)
    sparsity_list2 = [0 for x in range(0, 4)]
    for index in range(0, len(sparsity_list)):
        if sparsity_list[index] < 25:
            sparsity_list2[0] += 1
        elif sparsity_list[index] < 50:
            sparsity_list2[1] += 1
        elif sparsity_list[index] < 75:
            sparsity_list2[2] += 1
        elif sparsity_list[index] < 100:
            sparsity_list2[3] += 1
    # print("sparsity histogram")
    # print(sparsity_list2)
    
    total_sparsity_sum = 0
    for elem in sparsity_list:
        total_sparsity_sum += elem
    total_sparsity = total_sparsity_sum / len(sparsity_list)
    # print("total sparsity: ", total_sparsity)
    
    # write = "total sparsity: "+ str(total_sparsity) + "\n"
    # perf_file.write(write)
    
    # csv_file.writerow(["total sparsity", total_sparsity])
    
    # print()
    # print()
    
    # total_qubits = circuit.all_qubits()
    total_depth = len(circuit)
    print("total circuit depth: ", total_depth)
    
    # write = "total depth: "+ str(total_depth) + "\n"
    # perf_file.write(write)
    
    # csv_file.writerow(["total depth", total_depth])
    
    total_position = total_depth * len(circuit.all_qubits())
    total_pos_count = 0
    all_ops = circuit.findall_operations(lambda op:op)
    for i, op in all_ops:
        total_pos_count += len(op.qubits)
    gate_density = (total_pos_count / total_position) * 100
    print("gate density: ", gate_density, " %")
    
    # write = "gate density: "+ str(gate_density) + "\n"
    # perf_file.write(write)
    
    # csv_file.writerow(["gate density", gate_density])
    
    # print()
    # print()
    
    
    
    
    
        
program_start = time.time()
    


gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=CHOOSE_API, verbosity=0, max_fused_gate_size=MAX_FUSED_QSIM)
# gpu_options = qsimcirq.QSimOptions(use_gpu=False, gpu_mode=0, verbosity=0, max_fused_gate_size=2)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

ql = cirq.LineQubit.range(30)
bench_circuit = cirq.Circuit(
    cirq.X(ql[0]), cirq.H(ql[1]), cirq.Z(ql[2]),  cirq.Y(ql[1]),cirq.CZ(ql[0], ql[1]), cirq.CNOT(ql[0], ql[1]), cirq.Z(ql[1]),  cirq.T(ql[1]), cirq.H(ql[1])
)

# kraus = cirq.kraus(bench_circuit)
# print(cirq.unitary(bench_circuit))
# print(kraus)
# noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.2))

# file = open('/workspace/examples/Benchmark/combinational/grover/grover_17.qasm',mode='r')
# file = open('/workspace/examples/QASMBench/small/qaoa_n3/qaoa_n3.qasm',mode='r')
# file = open('/workspace/examples/QASMBench/medium/qft_n18/qft_n18.qasm',mode='r')
# file = open('/workspace/examples/QASMBench/medium/dnn_n16/dnn_n16.qasm',mode='r')
# file = open('/workspace/examples/QASMBench/large/qft_n29/qft_n29.qasm',mode='r')
# file = open('/workspace/examples/QASMBench/medium/ising_n26/ising_n26.qasm',mode='r')


bench_circuit = circuit_from_qasm(qasm_input)

extended_bench_circuit = bench_circuit
repetition = 0
for index in range(repetition):
    extended_bench_circuit.append(bench_circuit)



max_cluster_degree = MAX_FUSED_BLINQ
result_circuit = apply_BlinQ_module(extended_bench_circuit, max_cluster_degree)
circuit_analysis(result_circuit, max_cluster_degree)

# result_circuit = apply_BlinQ(extended_bench_circuit, max_cluster_degree)
# result_circuit = bench_circuit

# print("Circuit analysis: original")
# print(extended_bench_circuit)
# circuit_analysis(extended_bench_circuit, max_cluster_degree)

# print("Circuit analysis: optimized")
# print(result_circuit)
# circuit_analysis(result_circuit, max_cluster_degree)


# print("Original      Circuit:", bench_circuit, "\n", sep="\n")
# print("result        Circuit:", result_circuit, "\n", sep="\n")
# print("Validation for final         merge: ", validate_circuit(bench_circuit, result_circuit))
    
# q0, q1, q2 = cirq.LineQubit.range(3)
# print(cirq.unitary(cirq.Circuit(cirq.X(q0), cirq.H(q1))))
# circuit = cirq.Circuit(cirq.CNOT(q0, q2), cirq.H(q1))
# op1 = cirq.CNOT(q0, q2)
# op1_qubits = [qubits for qubits in op1.qubits]
# print(op1_qubits)

# q0, q1, q2, q3, q4, q5 = cirq.LineQubit.range(6)
# print(np.kron(cirq.unitary(cirq.H(q0)), cirq.unitary(cirq.H(q1))))
# print(cirq.unitary(cirq.CCX(q0, q2, q1)))
# circuit = cirq.Circuit(cirq.H(q2),cirq.CZ(q0, q2),cirq.H(q2),cirq.CZ(q1, q2) )
# circuit2 = cirq.Circuit(cirq.X(q0), cirq.H(q1))
# print(cirq.unitary(circuit))
# print(circuit)
# print(cirq.unitary(extended_bench_circuit))
# print("Validation for compare: ", validate_circuit(extended_bench_circuit, circuit))
# print(circuit2)
# for moment in bench_circuit:
#     for op in moment:
#         print(op)
#         print(cirq.unitary(op))
#         print(cirq.unitary(moment))

# perf_file.close()
# perf_file2.close()

# program_stop = time.time() - program_start
# print(f'Program runtime: {program_stop} seconds.')

# print()
