from parse import compile
import sys
import latextable

from tabulate import tabulate

######## usage format
######## algo_name, start_qubit, stop_qubit, stride, api_option, MAX_FUSED_QSIM, MAX_FUSED_BLINQ, qsim or blinq
######## python3 get_circuit_table.py bv 10 32 2 custatevec 2 4 blinq

### file size
# bv: 10 32 2
# adder 10 31 3
# grover: 11 31 2

# 회로            총 게이트 수, 총 depth, 총 sparsity, 총 gate density. 1: 게이트 수, avg. gate execution time, avg. sparsity. 2, 3, 4


bench_name = sys.argv[1]
input_data_file_prefix = "./blinq_result/"+bench_name+"/result/"+bench_name+"_"
qubit_start = int(sys.argv[2])
qubit_stop = int(sys.argv[3])
qubit_stride = int(sys.argv[4])



execution_data_list = []
data_list = []


parse_gate_execution_time = compile("time consumed per gate: {} mseconds\n")
parse_gate_index_num = compile("qubit size: {}\n")
parse_gate_sparsity = compile("sparsity: {}\n")
        

for qubit_size in range(qubit_start, qubit_stop+1, qubit_stride):
    input_data_file=input_data_file_prefix+str(qubit_size)+"_"+sys.argv[5]+"_"+sys.argv[6]+"_"+sys.argv[7]+"_"+sys.argv[8]
    
    with open(input_data_file) as f:
        lines = f.readlines()

    circuit_depth_line=lines[0]
    gate_density_line=lines[1]
    
    parse_circuit_depth = compile("total circuit depth:  {}\n")
    parse_gate_density = compile("gate density:  {}  %\n")
    
    index_num_list = [0] * 7
    sparsity_list = [0] * 7
    gate_execution_time_list = [0] * 7
    
    gate_num = int((len(lines)-2)/5)
    
    for line_index in range(gate_num):
        index_num = int(parse_gate_index_num.parse(lines[line_index*5+3])[0])
        
        index_num_list[index_num] += 1
        sparsity_list[index_num] += float(parse_gate_sparsity.parse(lines[line_index*5+4])[0])
        gate_execution_time_list[index_num] += float(parse_gate_execution_time.parse(lines[(line_index+1)*5])[0])
        
    sparsity_avg = 0
    
    for qubit_num in range(7):
        sparsity_avg += sparsity_list[qubit_num]
        if index_num_list[qubit_num] != 0:
            sparsity_list[qubit_num] /= index_num_list[qubit_num]
            gate_execution_time_list[qubit_num] /= index_num_list[qubit_num]
            
    sparsity_avg /= gate_num
    
    circuit_depth=int(parse_circuit_depth.parse(circuit_depth_line)[0])
    gate_density=float(parse_gate_density.parse(gate_density_line)[0])
    
    
# print("avg sparsity: ", sparsity_avg)
print("circuit_depth: ", circuit_depth)
print("gate_density: ", gate_density)
print("gate_num: ", gate_num)
print("index_num_list", index_num_list)

# 회로            총 게이트 수, 총 depth, 총 sparsity, 총 gate density. 1: 게이트 수, avg. gate execution time, avg. sparsity. 2, 3, 4

# gate_num
# circuit_depth
# gate_density
# sparsity_avg

# index_num_list[7]
# gate_execution_time_list[7]
# sparsity_list[7]



rows = [['\makecell{\# of Total \\\Gates}', '\makecell{Circuit\\\Depth}', '\makecell{Gate\\\Density}', '\makecell{Average\\\Sparsity}', \
        # '1Q num gate', '1Q exec time', '1Q sparsity', \
        # '2Q num gate', '2Q exec time', '2Q sparsity', \
        '\makecell{\# 3-Qubit\\\Gate}', '\makecell{3-Qubit\\\Exec. Time}', '\makecell{3-Qubit\\\Sparsity}', \
        '\makecell{\# 4-Qubit\\\Gate}', '\makecell{4-Qubit\\\Exec. Time}', '\makecell{4-Qubit\\\Sparsity}'],
        # '4Q num gate', '4Q exec time', '4Q sparsity', ],
        [gate_num, circuit_depth, gate_density, sparsity_avg,\
        # index_num_list[1], gate_execution_time_list[1], sparsity_list[1],\
        # index_num_list[2], gate_execution_time_list[2], sparsity_list[2],\
        index_num_list[3], gate_execution_time_list[3], sparsity_list[3],\
        index_num_list[4], gate_execution_time_list[4], sparsity_list[4]]]


print('\nTexttable Latex:')
print(latextable.draw_latex(rows, caption="GTOQC Result Summary"))
