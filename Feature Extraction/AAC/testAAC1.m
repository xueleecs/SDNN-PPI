clc;
close all;
clear all;
load('N_protein_A.mat')
%[x,y]=fastaread('SC22000.txt');
% txt=y';

f=N_protein_A;
% f=txt;

strn = regexprep(f,'[XBJOUZ]','');
k=length(strn);
for i=1:k
    q(i)=length(strn{i});
    a(i)=aacount(strn{i});
end
ch='ARNDCQEGHILKMFPSTWYV';
for j=1:k
for i=1:20
    AAC1(j,i)=a(j).(ch(i))/q(j);
end
end

%save AAC1;

csvwrite('AAC_N1.csv', AAC1)
% csvwrite('AAC_N2.csv', AAC1)
% csvwrite('AAC_P1.csv', AAC1)
% csvwrite('AAC_P2.csv', AAC1)

