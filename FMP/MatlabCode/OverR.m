function y = OverR(x1,x2)
xstart = min(min(x1(:,1)),min(x2(:,1)));
ystart = min(min(x1(:,2)),min(x2(:,2)));
xend = max(max(x1(:,1)),max(x2(:,1)));

// overlapping area width and height
width = (max(x1(:,1))-min(x1(:,1))) + (max(x2(:,1))-min(x2(:,1))) - (xend-xstart);
height = (max(x1(:,2))-min(x1(:,2))) + (max(x2(:,2))-min(x2(:,2))) - (yend-ystart);


if width<=0 || height<= 0
    y = 0;
else
    area = width*height;
    area1 = (max(x1(:,1))-min(x1(:,1)))*(max(x1(:,2))-min(x1(:,2)));
    y = area/area1;
end