function [mcd1po]=MCD1D(xulie)
% load MCDNE
% load MCDPO;
% xulie=numberpo;
% xulie=numberpo;
num=numel(xulie);
cheng=[];
for j=1:num
    set=xulie{j};
    N=length(set);
for i=1:7
    a{i}=length(find(set==i));
end
M=cell2mat(a);
MC1{j}=M/N;
a=[];
set=[];
end
% mcd1ne=MC1;
% NUM=numel(MC1);
for k=1:num
    cheng=[cheng,[MC1{k}]];
end
mcd1po=cheng;
% save MCD1PO.mat mcd1po
