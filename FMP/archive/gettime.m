function y = gettime(x,w)

[t,l] = size(x);
g = floor(l/w);
n = 1;

for i = 1:t-1
    x1 = x(i,:);
    for j = i+1:t
        x2 = x(j,:);
        for k = 1:g
            t1 = [min(x1((k-1)*w+1:k*w)),max(x1((k-1)*w+1:k*w))];
            t2 = [min(x2((k-1)*w+1:k*w)),max(x2((k-1)*w+1:k*w))];
            [T1,T2] = meshgrid(t1,t2);
            y(n,:,(k-1)*2+1:k*2) = [T1(:),T2(:)];
        end
        n = n+1;
    end
end

end