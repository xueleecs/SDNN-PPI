function [mark] = MCDexchange( choice)
h1=0;h2=0;h3=0;h4=0;h5=0;h6=0;h7=0;h8=0;h9=0;h10=0;
h11=0;h12=0;h13=0;h14=0;h15=0;h16=0;h17=0;h18=0;h19=0;h20=0;h21=0;
switch choice
    case '1  2'
        h1=1;
    case '2  1'
        h1=1;
    case '3  1'
        h2=1;
    case '1  3'
        h2=1;
    case '1  4'
        h3=1;
    case '4  1'
        h3=1;
    case '5  1'
        h4=1;
    case '1  5'
        h4=1;
    case '1  6'
        h5=1;
    case '6  1'
        h5=1;
    case '1  7'
        h6=1;
    case '7  1'
        h6=1;    
    case '2  3'
        h7=1;
    case '3  2'
        h7=1;
    case '2  4'
        h8=1;
    case '4  2'
        h8=1;
    case '2  5'
        h9=1;
    case '5  2'
        h9=1;
    case '2  6'
        h10=1;
    case '6  2'
        h10=1;
    case '3  4'
        h11=1;
    case '4  3'
        h11=1;
    case '3  5'
        h12=1;
    case '5  3'
        h12=1;
    case '3  6'
        h13=1;
    case '6  3'
        h13=1;
    case '3  7'
        h14=1;
    case '7  3'
        h14=1;
    case '4  5'
        h15=1;
    case '5  4'
        h15=1;
    case '4  6'
        h16=1;
    case '6  4'
        h16=1; 
    case '4  7'
        h17=1;
    case '7  4'
        h17=1;
    case '5  6'
        h18=1;
    case '6  5'
        h18=1;
    case '5  7'
        h19=1;
    case '7  5'
        h19=1;
    case '6  7'
        h20=1;
    case '7  6'
        h20=1;
    case '2  7'
        h21=1;
    case '7  2'
        h21=1;    
end
mark1=[h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20,h21];
mark2=find(mark1==1);
mark3=isempty(mark2);
if (mark3==1);
    mark=0;
else
    mark=mark2;
end
    
end

