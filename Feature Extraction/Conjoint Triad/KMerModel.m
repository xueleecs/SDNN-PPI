function [Kmer,maxmin_comp]= KMerModel(seqname,kmer)
% in='input the K-mer: 2-Dime, 3-Trimer,4-Quadmer';
% N=input(in);
% ss=('MSAQAELSREENVYMAKLAEQAERYEEMVEFMEKVAKTVDSEELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEESRGNEDRVTLIKDYRGKIETELTKICDGILKLLESHLVPSSTAPESKVFYLKMKGDYYRYLAEFKTGAERKDAAENTMVAYKAAQDIALAELPPTHPIRLGLALNFSVFYYEILNSPDRACNLAKQAFDEAISELDTLSEESYKDSTLIMQLLRDNLTLWTSDISEDTAEEIREAPKRDSSEGQ');

N=kmer; 
ss=seqname;
%%%%%%%%%%%%%%%%%%%%%Di-Mer%%%%%%%%%%%%%%%%%%%%%%%%%
if N==2
% N=2;
amino={'1', '2', '3', '4', '5', '6', '7'};
for i=1:7
   for k=1:7
       dimer{i,k}=[amino{i},amino{k}];
   end
end
twomer=reshape(dimer,1,7^N);

aminoclass=Aminogrp(ss);
pname=aminoclass;
seq=pname;

Nmer1=nmercount(seq,N);  %%%%%%%% Count no. of dipeptides
Nmer=Nmer1(:,1);Dipep1=zeros(1,7^N);
for i=1:length(twomer)
    dd=strmatch(twomer(i),Nmer);
    if dd~=0
    Dipep1(i)=Nmer1{dd,2};
    else Dipep1(i)=0;
    end
end
Kmer=Dipep1./(sum(Dipep1));
%maxmin_comp=(Dipep1-min(Dipep1))./(max(Dipep1)-min(Dipep1));
%%%%%%%%%%%%%%%%%%%%%%%%%Tri-Mer%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif N==3
%N=3;
amino={'1', '2', '3', '4', '5', '6', '7'};
for i=1:7
   for k=1:7
       for j=1:7
       trimer{i,k,j}=[amino{i},amino{k},amino{j}];
       end
   end
end
threemer=reshape(trimer,1,7^N);

aminoclass=Aminogrp(ss);
pname=aminoclass;
seq=pname;

Nmer1=nmercount(seq,N);  %%%%%%%% Count no. of dipeptides
Nmer=Nmer1(:,1);tripep1=zeros(1,7^N);
for i=1:length(threemer)
    dd=strmatch(threemer(i),Nmer);
    if dd~=0
    tripep1(i)=Nmer1{dd,2};
    else tripep1(i)=0;
    end
end
Kmer=tripep1./(sum(tripep1));
%maxmin_comp=(tripep1-min(tripep1))./(max(tripep1)-min(tripep1));
%%%%%%%%%%%%%%%%%%%%%%%%% Quad-Mer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif N==4
%N=4;
amino={'1', '2', '3', '4', '5', '6', '7'};
for i=1:7
   for k=1:7
       for j=1:7
           for m=1:7
       quadmer{i,k,j,m}=[amino{i},amino{k},amino{j},amino{m}];
           end
       end
   end
end
fourmer=reshape(quadmer,1,7^N);
aminoclass=Aminogrp(ss);
pname=aminoclass;
seq=pname;
Nmer1=nmercount(seq,N);  %%%%%%%% Count no. of quadmers
Nmer=Nmer1(:,1);quadpep1=zeros(1,7^N);
for i=1:length(fourmer)
    dd=strmatch(fourmer(i),Nmer);
    if dd~=0
    quadpep1(i)=Nmer1{dd,2};
    else quadpep1(i)=0;
    end
end
Kmer=quadpep1./(sum(quadpep1));
%maxmin_comp=(quadpep1-min(quadpep1))./(max(quadpep1)-min(quadpep1));
%%%%%%%%%%%%%%%%%%%%%% Penta-Mer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif N==5
%N=4;
amino={'1', '2', '3', '4', '5', '6', '7'};
for i=1:7
   for k=1:7
       for j=1:7
           for m=1:7
               for n=1:7
       pentamer{i,k,j,m,n}=[amino{i},amino{k},amino{j},amino{m},amino{n}];
               end
           end
       end
   end
end
fivemer=reshape(pentamer,1,7^N);
aminoclass=Aminogrp(ss);
pname=aminoclass;
seq=pname;
Nmer1=nmercount(seq,N);  %%%%%%%% Count no. of quadmers
Nmer=Nmer1(:,1);pentapep1=zeros(1,7^N);
for i=1:length(fivemer)
    dd=strmatch(fivemer(i),Nmer);
    if dd~=0
    pentapep1(i)=Nmer1{dd,2};
    else pentapep1(i)=0;
    end
end
Kmer=pentapep1./(sum(pentapep1));
%maxmin_comp=(pentapep1-min(pentapep1))./(max(pentapep1)-min(pentapep1));
end

