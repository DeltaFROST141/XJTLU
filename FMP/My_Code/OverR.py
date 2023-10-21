import numpy as np

def OverR(x1, x2):
    xstart = min(np.min(x1[:, 0]), np.min(x2[:, 0]))
    ystart = min(np.min(x1[:, 1]), np.min(x2[:, 1]))
    xend = max(np.max(x1[:, 0]), np.max(x2[:, 0]))
    yend = max(np.max(x1[:, 1]), np.max(x2[:, 1]))

    # ! width and height of overlapping area
    width = (np.max(x1[:, 0]) - np.min(x1[:, 0])) + (np.max(x2[:, 0]) - np.min(x2[:, 0])) - (xend - xstart)
    height = (np.max(x1[:, 1]) - np.min(x1[:, 1])) + (np.max(x2[:, 1]) - np.min(x2[:, 1])) - (yend - ystart)

    # ? why using area/area1 ? Not the area2 or the whole area ?
    if width <= 0 or height <= 0:
        return 0
    else:
        area = width * height
        area1 = (np.max(x1[:, 0]) - np.min(x1[:, 0])) * (np.max(x1[:, 1]) - np.min(x1[:, 1]))
        print('The overlapping area is {}'.format(area))
        print(area1)
        return area / area1

# ! the first point is left lower corner and the second point is right upper corner 
x1 = np.array([[2, 1], [7, 5]])
x2 = np.array([[1, 1], [4, 8]])
print('The ratio of area is {}'.format(OverR(x1, x2)))
