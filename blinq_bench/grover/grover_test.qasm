OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
h q[5];
//cx q[1], q[0];
//ccx q[0],q[1],q[4];
ccx q[2],q[4],q[3];
//ccx q[0],q[1],q[4];
