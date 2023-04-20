import subprocess
from optparse import OptionParser

######## usage format
######## use_gpu, bench, cuda_implmentation, run_blinq, MAX_FUSED_BLINQ, num_qubits
######## python3 run_blinq.py -g True -b bv -c cuda -r True -f 2 --start 10 --stop 10 --stride 2

### file size
# bv: 10 32 2
# adder 10 31 3
# grover: 11 31 2
# pe: 10 30 2
# qft: 10 32 2

parser = OptionParser()
parser.add_option("-g", "--use_gpu", dest="use_gpu", default=True, help="True for GPU usage")
parser.add_option("-b", "--bench", dest="bench", help="set benchmark name")
parser.add_option("-c", "--cuda_implementation", dest="cuda_implementation", help="set cuda kernel to cuda or custatevec")
parser.add_option("-r", "--run_blinq", dest="run_blinq", help="True if BlinQ is used")
parser.add_option("-f", "--MAX_FUSED_BLINQ", dest="MAX_FUSED_BLINQ", help="blinq_max_fused number")
parser.add_option("--start", dest="qubit_start", help="start qubit number")
parser.add_option("--stop", dest="qubit_stop", help="ending qubit number")
parser.add_option("--stride", dest="qubit_stride", help="distance of qubit number")

(options, args) = parser.parse_args()

use_gpu=options.use_gpu
bench=options.bench
cuda_implementation=options.cuda_implementation
run_blinq=options.run_blinq
MAX_FUSED_BLINQ=options.MAX_FUSED_BLINQ
qubit_start = int(options.qubit_start)
qubit_stop = int(options.qubit_stop)
qubit_stride = int(options.qubit_stride)

if run_blinq=="True":
    run_blinq_bool=True
else:
    run_blinq_bool=False

if run_blinq_bool:
    backend="blinq"
else:
    backend="qsim"

for num_qubits in range(qubit_start, qubit_stop+1, qubit_stride):
    if run_blinq_bool:
        output_file = "./blinq_result/"+bench+"/data/"+str(num_qubits)+"_"+cuda_implementation+"_"+backend+"_"+MAX_FUSED_BLINQ
    else:
        output_file = "./blinq_result/"+bench+"/data/"+str(num_qubits)+"_"+cuda_implementation+"_"+backend

    command = "python3 blinq_auto.py "+"-g "+use_gpu+" -b "+bench+" -c "+cuda_implementation+" -r "+run_blinq+" -f "+MAX_FUSED_BLINQ+" -q "+str(num_qubits)+" > " + output_file
    subprocess.run(command, shell=True)
    
    
    