function y = getgradient(x,w)

[t,l] = size(x);
g = floor(l/w);
n = 1;

mi = min(min(x));
mx = max(max(x));

for i = 1:t
    x1 = x(i,:);
    for k = 1:g
        grad(i,k,:) = gradient(x1((k-1)*w+1:k*w));
    end
end

grad = reshape(mapminmax(reshape(grad,1,[]),mi,mx),t,g,[]);

for i = 1:t-1
    grad1 = squeeze(grad(i,:,:));
    for j = i+1:t
        grad2 = squeeze(grad(j,:,:));
        for k = 1:g
            g1 = [min(grad1(k,:)),max(grad1(k,:))];
            g2 = [min(grad2(k,:)),max(grad2(k,:))];
            [G1,G2] = meshgrid(g1,g2);
            y(n,:,(k-1)*2+1:k*2) = [G1(:),G2(:)];
        end
        n = n+1;
    end
end

end