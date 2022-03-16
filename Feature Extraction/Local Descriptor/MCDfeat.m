clear all
clc
% [x,y]=fastaread('SC22000.txt');
load('N_protein_A.mat')
txt=N_protein_A;
strn = regexprep(txt,'[XBJOUZ]','');
num1=numel(strn);
MCD_Pa=[];

for i=1:num1
[M1]=MCDZD(strn{i});
M=[M1];
MCD_Pa=[MCD_Pa;M];
clear M;clear M1;
end
csvwrite('LD_N1.csv', MCD_Pa)
% csvwrite('LD_N2.csv', MCD_Pa)
% csvwrite('LD_P1.csv', MCD_Pa)
% csvwrite('LD_P2.csv', MCD_Pa)