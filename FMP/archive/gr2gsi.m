function y = gr2gsi(g1,x)
g = size(g1,2);
h = 1/x;
gd = 0:h:1;

for i = 1:x
    gx = gd([i,i+1]);
    for j = 1:x
        gy = gd([j,j+1]);
        [X,Y] = meshgrid(gx,gy);
        gc = [X(:),Y(:)];
        for k = 1:g/2
            r(k) = OverR(gc,g1(:,(k-1)*2+1:k*2));
        end
        y(i,j) = sum(r);
    end
end

y = fliplr(y);
y = y';