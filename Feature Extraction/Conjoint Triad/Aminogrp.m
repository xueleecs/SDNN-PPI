function [aminoclass] = Aminogrp(ss)

%ss=('MSAQAELSREENVYMAKLAEQAERYEEMVEFMEKVAKTVDSEELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEESRGNEDRVTLIKDYRGKIETELTKICDGILKLLESHLVPSSTAPESKVFYLKMKGDYYRYLAEFKTGAERKDAAENTMVAYKAAQDIALAELPPTHPIRLGLALNFSVFYYEILNSPDRACNLAKQAFDEAISELDTLSEESYKDSTLIMQLLRDNLTLWTSDISEDTAEEIREAPKRDSSEGQ');
char_seq=ss;
len = length(char_seq);
   for i = 1:len
      switch upper(char_seq(i))
         case 'L'
            aminoclass(i) = '2';
         case 'I'
            aminoclass(i) = '2';
         case 'N'
            aminoclass(i) = '4';
         case 'G'
            aminoclass(i) = '1';
         case 'V'
            aminoclass(i) = '1';
         case 'E'
            aminoclass(i) = '6';
         case 'P'
            aminoclass(i) = '2';
         case 'H'
            aminoclass(i) = '4';
         case 'K'
            aminoclass(i) = '5';
         case 'A'
            aminoclass(i) = '1';
         case 'Y'
            aminoclass(i) = '3';
         case 'W'
            aminoclass(i) ='4';
         case 'Q'
            aminoclass(i) = '4';
         case 'M'
            aminoclass(i) = '3';
         case 'S'
            aminoclass(i) ='3';
         case 'C'
            aminoclass(i) = '7';
         case 'T'
            aminoclass(i) = '3';
         case 'F'
            aminoclass(i) = '2';
         case 'R'
            aminoclass(i) = '5';
         case 'D'
            aminoclass(i) = '6';
%          otherwise
%             error('ERROR! INVALID AMINO ACID FOUND! CHECK CHAR. STRING!')
%             beep on, beep
      end
   end

%aminoclass = aminoclass(:)';