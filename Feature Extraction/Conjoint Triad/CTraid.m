clc;
clear all;
close all;

% [x,y]=fastaread('SC22000.txt');
% txt=y';
% f=txt;
load('N_protein_A.mat')

f=N_protein_A;
strn = regexprep(f,'[XBJOUZ]','');
k=length(strn);

mer=3;
len3=k;
for i=1:len3
   [Trimer_Train(i,:)]=KMerModel(strn{i},mer);
   
end
csvwrite('CT_N1.csv', Trimer_Train)
% csvwrite('CT_N2.csv', Trimer_Train)
% csvwrite('CT_P1.csv', Trimer_Train)
% csvwrite('CT_P2.csv', Trimer_Train)