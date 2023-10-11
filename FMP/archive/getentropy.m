function y = getentropy(x,w)
[t,l] = size(x);
g = floor(l/w);
n = 1;

mi = min(min(x));
mx = max(max(x));

for i = 1:t
    x1 = x(i,:);
    for k = 1:g
        entro(i,k,:) = pea(x1((k-1)*w+1:k*w),3,1);
    end
end

entro = reshape(mapminmax(reshape(entro,1,[]),mi,mx),t,g,[]);

for i = 1:t-1
    entro1 = squeeze(entro(i,:,:));
    for j = i+1:t
        entro2 = squeeze(entro(j,:,:));
        for k = 1:g
            e1 = [min(entro1(k,:)),max(entro1(k,:))];
            e2 = [min(entro2(k,:)),max(entro2(k,:))];
            [E1,E2] = meshgrid(e1,e2);
            y(n,:,(k-1)*2+1:k*2) = [E1(:),E2(:)];
        end
        n = n+1;
    end
end

end