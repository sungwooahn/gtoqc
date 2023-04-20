# %%
from parse import compile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from optparse import OptionParser


######## usage format
######## python3 get_compute_figure.py -b bv -c cuda 

### file size
# bv: 10 32 2
# adder 10 31 3
# grover: 11 31 2
# pe: 10 30 2
# qft: 10 32 2



parser = OptionParser()
parser.add_option("-b", "--bench", dest="bench", help="set benchmark name")
parser.add_option("-c", "--cuda_implementation", dest="cuda_implementation", help="set cuda kernel to cuda or custatevec")

(options, args) = parser.parse_args()

bench=options.bench
cuda_implementation=options.cuda_implementation

bench_name = bench
input_data_file_prefix = "./blinq_result/"+bench_name+"/data/"

if bench=='bv':
    qubit_start=10
    qubit_stop=32
    qubit_stride=2
elif bench=='adder':
    qubit_start=10
    qubit_stop=31
    qubit_stride=3
elif bench=='grover':
    qubit_start=11
    qubit_stop=31
    qubit_stride=2
elif bench=='pe':
    qubit_start=10
    qubit_stop=30
    qubit_stride=2
else:
    qubit_start=10
    qubit_stop=32
    qubit_stride=2

qubit_index = []
exec_time=[]        

for num_qubits in range(qubit_start, qubit_stop+1, qubit_stride):
    input_data_file=input_data_file_prefix+str(num_qubits)+"_"+cuda_implementation+"_"+"qsim"
    
    with open(input_data_file) as f:
        lines = f.readlines()

    total_exec_time_line=lines[-1]
    parse_execution_time = compile("time consumed total: {} mseconds\n")
        
    qubit_index.append(num_qubits)
    exec_time.append(float(parse_execution_time.parse(total_exec_time_line)[0]))

qsim_exec_time=exec_time

cd_exec_time=[]
for cd in range(2, 7, 1):
    exec_time=[]
    for num_qubits in range(qubit_start, qubit_stop+1, qubit_stride):
        input_data_file=input_data_file_prefix+str(num_qubits)+"_"+cuda_implementation+"_blinq_"+str(cd)

        with open(input_data_file) as f:
            lines = f.readlines()

        total_exec_time_line=lines[-1]
        parse_execution_time = compile("time consumed total: {} mseconds\n")

        exec_time.append(float(parse_execution_time.parse(total_exec_time_line)[0]))
        
    cd_exec_time.append(exec_time)

cd_2_exec_time = cd_exec_time[0]
cd_3_exec_time = cd_exec_time[1]
cd_4_exec_time = cd_exec_time[2]
cd_5_exec_time = cd_exec_time[3]
cd_6_exec_time = cd_exec_time[4]

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

exec_time_ratio=[]
cd_2_exec_time_ratio=[]
cd_3_exec_time_ratio=[]
cd_4_exec_time_ratio=[]
cd_5_exec_time_ratio=[]
cd_6_exec_time_ratio=[]

fig2 = plt.figure(figsize=(10,4))
ax2 = plt.axes()
ax2.xaxis.set_major_locator(MultipleLocator(2))
# ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(40))
ax2.yaxis.set_minor_locator(MultipleLocator(20))
ax2.grid(True,which='major',axis='both',alpha=0.3)
         
font1 = {'weight': 'bold',
         'size': 12
         }
plt.xlabel('Number of Qubits', labelpad=5, fontdict=font1)
plt.ylabel('Normalized Performance (%)',labelpad=5, fontdict=font1)


qubit_count = int((qubit_stop-qubit_start)/qubit_stride)+1

for index in range(qubit_count):
    exec_time_ratio.append((qsim_exec_time[index]-qsim_exec_time[index])/qsim_exec_time[index]*100)
    cd_2_exec_time_ratio.append((qsim_exec_time[index]-cd_2_exec_time[index])/qsim_exec_time[index]*100)
    cd_3_exec_time_ratio.append((qsim_exec_time[index]-cd_3_exec_time[index])/qsim_exec_time[index]*100)
    cd_4_exec_time_ratio.append((qsim_exec_time[index]-cd_4_exec_time[index])/qsim_exec_time[index]*100)
    cd_5_exec_time_ratio.append((qsim_exec_time[index]-cd_5_exec_time[index])/qsim_exec_time[index]*100)
    cd_6_exec_time_ratio.append((qsim_exec_time[index]-cd_6_exec_time[index])/qsim_exec_time[index]*100)

ax2.plot(qubit_index,exec_time_ratio,marker='o',markersize=8,markerfacecolor='orange',color='orange',markeredgecolor='black', markeredgewidth=2, linestyle='--',linewidth=3,label='qsim line')
ax2.plot(qubit_index,cd_2_exec_time_ratio,marker='o',markersize=8,markerfacecolor='darkviolet',color='darkviolet',markeredgecolor='black',markeredgewidth=2,linestyle='--',linewidth=3,label='cd_2 line')
ax2.plot(qubit_index,cd_3_exec_time_ratio,marker='o',markersize=8,markerfacecolor='royalblue',color='royalblue',markeredgecolor='black',markeredgewidth=2,linestyle='--',linewidth=3,label='cd_3 line')
ax2.plot(qubit_index,cd_4_exec_time_ratio,marker='o',markersize=8,markerfacecolor='darkgoldenrod',color='darkgoldenrod',markeredgecolor='black',markeredgewidth=2,linestyle='--',linewidth=3,label='cd_4 line')
ax2.plot(qubit_index,cd_5_exec_time_ratio,marker='o',markersize=8,markerfacecolor='green',color='green',markeredgecolor='black',markeredgewidth=2,linestyle='--',linewidth=3,label='cd_5 line')
ax2.plot(qubit_index,cd_6_exec_time_ratio,marker='o',markersize=8,markerfacecolor='crimson',color='crimson',markeredgecolor='black',markeredgewidth=2,linestyle='--',linewidth=3,label='cd_6 line')
ax2.legend()

save_file="./blinq_result/"+bench_name+"/"+bench_name+"_"+cuda_implementation+"_compute.png"
ax2.figure.savefig(save_file, bbox_inches='tight')



# %%
