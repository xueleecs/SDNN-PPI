% clear all
% clc
% load F
% label=length(fast1);
% fastmat=cell2mat(F);
% set=[];
% mat1=fastmat(1:label);
% mat2=fastmat(label+1:end);
% clear all
% clc
%  load MCDNE
function [mcd2ne]=MCD2D(numberne)
lam=numel(numberne);
number=numberne;
cheng=[];
for j=1:lam
set=number{j};
choice=set;
% choice=[1;2;3;2;2;1];
% choice=choice';
Zmark=[];
for i=1:length(choice)-1;
choice1=choice(:,[i,i+1]);
choice2=num2str(choice1);
% a='12';
% h=[];
mark=MCDexchange(choice2);
Zmark=[Zmark,mark];
end
N1=length(choice);
for k=1:21
    ve{k}=length(find(Zmark==k));
end
MMCD1=cell2mat(ve);
MMCD2=MMCD1/N1;
vector1{j}=MMCD2;
ve=[];
set=[];
end
NUM=numel(vector1);
% for k=1:2:(NUM-1)
%     cheng=[cheng;[vector1{k},vector1{k+1}]];
% end
for k=1:lam
    cheng=[cheng,[vector1{k}]];
end
mcd2ne=cheng;
% save MCD2NE.mat mcd2ne

% switch choice2
%     case '1  2'
%         h1=1;
%     case '2  1'
%         h1=1;
%     case '3 1'
%         h2=1;
%     case '1 3'
%         h2=1;
% end
% end
% mark=[h1,h2];