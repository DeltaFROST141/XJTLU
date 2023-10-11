def get_overlap(x_1, x_2):
    x_start = min(min(x_1[:, 0], x_2[:, 0]))
    y_start = min(min(x_1[:, 1], x_2[:, 1]))
    x_end = max(max(x_1[:, 0], x_2[:, 0]))
    y_end = max(max(x_1[:, 1], x_2[:, 1]))
    width = max(x_1[:, 0]) - min(x_1[:, 0]) + max(x_1[:, 0]) - min(
        x_1[:, 0]) - (x_end - x_start)
    height = max(x_2[:, 1]) - min(x_2[:, 1]) + max(x_2[:, 1]) - min(
        x_2[:, 1]) - (y_end - y_start)
    if width <= 0 or height <= 0:
        return 0
    else:
        area = width * height / (
            (max(x_1[:, 0]) - min(x_1[:, 0])) * (max(x_2[:, 1]) - min(x_2[:, 1])))
        return area
