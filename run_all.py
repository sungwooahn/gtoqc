import subprocess
import sys
from optparse import OptionParser


####### usage format
######## python3 run_all.py -g True 

### file size
# bv: 10 32 2
# adder 10 31 3
# grover: 11 31 2
# pe: 10 30 2
# qft: 10 32 2

parser = OptionParser()
parser.add_option("-g", "--use_gpu", dest="use_gpu", default=True, help="True for GPU usage")
parser.add_option("-b", "--bench", dest="bench", help="set benchmark name")

(options, args) = parser.parse_args()

use_gpu=options.use_gpu
bench=options.bench

# for algo in ['bv', 'adder', 'grover', 'pe', 'qft']:
# for algo in ['bv']:
for algo in ['adder', 'grover']:
    if algo=='bv':
        qubit_start='10'
        qubit_stop='32'
        qubit_stride='2'
    elif algo=='adder':
        qubit_start='10'
        qubit_stop='31'
        qubit_stride='3'
    elif algo=='grover':
        qubit_start='11'
        qubit_stop='31'
        qubit_stride='2'
    elif algo=='pe':
        qubit_start='10'
        qubit_stop='30'
        qubit_stride='2'
    else:
        qubit_start='10'
        qubit_stop='32'
        qubit_stride='2'
    
    
    for cuda_implementation in ['cuda', 'custatevec']:
        command = "python3 run_blinq_auto.py "+"-g "+use_gpu+" -b "+algo+" -c "+cuda_implementation+" -r False"\
                +" -f 2"+" --start "+qubit_start+" --stop "+qubit_stop+" --stride "+qubit_stride
        subprocess.run(command, shell=True)
        
        for max_fused in range(2, 7, 1):
            command = "python3 run_blinq_auto.py "+"-g "+use_gpu+" -b "+algo+" -c "+cuda_implementation+" -r True"\
                +" -f "+str(max_fused)+" --start "+qubit_start+" --stop "+qubit_stop+" --stride "+qubit_stride
            subprocess.run(command, shell=True)