OPENQASM 2.0;
include "qelib1.inc";
qreg q[96];
creg c[96];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
h q[30];
h q[31];
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
h q[39];
h q[40];
h q[41];
h q[42];
h q[43];
h q[44];
h q[45];
h q[46];
h q[47];
h q[48];
h q[49];
h q[50];
h q[51];
h q[52];
h q[53];
h q[54];
h q[55];
h q[56];
h q[57];
h q[58];
h q[59];
h q[60];
h q[61];
h q[62];
h q[63];
h q[64];
h q[65];
h q[66];
h q[67];
h q[68];
h q[69];
h q[70];
h q[71];
h q[72];
h q[73];
h q[74];
h q[75];
h q[76];
h q[77];
h q[78];
h q[79];
h q[80];
h q[81];
h q[82];
h q[83];
h q[84];
h q[85];
h q[86];
h q[87];
h q[88];
h q[89];
h q[90];
h q[91];
h q[92];
h q[93];
h q[94];
x q[95];
h q[95];
cx q[0],q[95];
cx q[1],q[95];
cx q[2],q[95];
cx q[3],q[95];
cx q[4],q[95];
cx q[5],q[95];
cx q[6],q[95];
cx q[7],q[95];
cx q[8],q[95];
cx q[9],q[95];
cx q[10],q[95];
cx q[11],q[95];
cx q[12],q[95];
cx q[13],q[95];
cx q[14],q[95];
cx q[15],q[95];
cx q[16],q[95];
cx q[17],q[95];
cx q[18],q[95];
cx q[19],q[95];
cx q[20],q[95];
cx q[21],q[95];
cx q[22],q[95];
cx q[23],q[95];
cx q[24],q[95];
cx q[25],q[95];
cx q[26],q[95];
cx q[27],q[95];
cx q[28],q[95];
cx q[29],q[95];
cx q[30],q[95];
cx q[31],q[95];
cx q[32],q[95];
cx q[33],q[95];
cx q[34],q[95];
cx q[35],q[95];
cx q[36],q[95];
cx q[37],q[95];
cx q[38],q[95];
cx q[39],q[95];
cx q[40],q[95];
cx q[41],q[95];
cx q[42],q[95];
cx q[43],q[95];
cx q[44],q[95];
cx q[45],q[95];
cx q[46],q[95];
cx q[47],q[95];
cx q[48],q[95];
cx q[49],q[95];
cx q[50],q[95];
cx q[51],q[95];
cx q[52],q[95];
cx q[53],q[95];
cx q[54],q[95];
cx q[55],q[95];
cx q[56],q[95];
cx q[57],q[95];
cx q[58],q[95];
cx q[59],q[95];
cx q[60],q[95];
cx q[61],q[95];
cx q[62],q[95];
cx q[63],q[95];
cx q[64],q[95];
cx q[65],q[95];
cx q[66],q[95];
cx q[67],q[95];
cx q[68],q[95];
cx q[69],q[95];
cx q[70],q[95];
cx q[71],q[95];
cx q[72],q[95];
cx q[73],q[95];
cx q[74],q[95];
cx q[75],q[95];
cx q[76],q[95];
cx q[77],q[95];
cx q[78],q[95];
cx q[79],q[95];
cx q[80],q[95];
cx q[81],q[95];
cx q[82],q[95];
cx q[83],q[95];
cx q[84],q[95];
cx q[85],q[95];
cx q[86],q[95];
cx q[87],q[95];
cx q[88],q[95];
cx q[89],q[95];
cx q[90],q[95];
cx q[91],q[95];
cx q[92],q[95];
cx q[93],q[95];
cx q[94],q[95];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
h q[30];
h q[31];
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
h q[39];
h q[40];
h q[41];
h q[42];
h q[43];
h q[44];
h q[45];
h q[46];
h q[47];
h q[48];
h q[49];
h q[50];
h q[51];
h q[52];
h q[53];
h q[54];
h q[55];
h q[56];
h q[57];
h q[58];
h q[59];
h q[60];
h q[61];
h q[62];
h q[63];
h q[64];
h q[65];
h q[66];
h q[67];
h q[68];
h q[69];
h q[70];
h q[71];
h q[72];
h q[73];
h q[74];
h q[75];
h q[76];
h q[77];
h q[78];
h q[79];
h q[80];
h q[81];
h q[82];
h q[83];
h q[84];
h q[85];
h q[86];
h q[87];
h q[88];
h q[89];
h q[90];
h q[91];
h q[92];
h q[93];
h q[94];
h q[95];
