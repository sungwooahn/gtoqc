
import sys

############usage
#### python3 patch.py vqe 28 27 1 

bench_name = sys.argv[1]

qubit_start = int(sys.argv[2])
qubit_stop = int(sys.argv[3])
qubit_stride = int(sys.argv[4])



# print("here")
for qubit_index in range(qubit_start, qubit_stop+1, qubit_stride):
    input_qasm_file = "./blinq_bench/"+bench_name+"/"+bench_name+"_"+str(qubit_index)+".qasm"
    output_qasm_file = "./blinq_bench/"+bench_name+"_patch/"+bench_name+"_"+str(qubit_index)+".qasm"

    with open(input_qasm_file) as f:
        lines = f.readlines()
        patch_lines = []

        for index in lines:
            check = index[0]+index[1]
            if check =="ba":
                continue
            elif check =="u1":
                
                parsed1 = index.split(" ")
                # print(parsed1)
                new_line = "s "+parsed1[1]
                patch_lines.append(new_line)
                continue
            else:
                patch_lines.append(index)
    f.close()   

    f = open(output_qasm_file,'w')
    for index in patch_lines:
        f.write(index)
    f.close()