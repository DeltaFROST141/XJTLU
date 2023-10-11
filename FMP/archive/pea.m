function pe = pea(y,m,t)
%  Calculate the permutation entropy
%  Input:   y: time series;
%           m: order of permuation entropy
%           t: delay time of permuation entropy, 
% Output: 
%           pe:    permuation entropy
ly = length(y);
permlist = perms(1:m);
c(1:length(permlist))=0;
    
 for j=1:ly-t*(m-1)
     [~,iv]=sort(y(j:t:j+t*(m-1)));
     for jj=1:length(permlist)
         if (abs(permlist(jj,:)-iv))==0
             c(jj) = c(jj) + 1 ;
         end
     end
 end
%c=c(c~=0);
p = c/sum(c);
for  i = 1:length(c)
    if c(i) == 0
        pe(i) = 0;
    else
        pe(i) = -(p(i)*log2(p(i)));
    end
end