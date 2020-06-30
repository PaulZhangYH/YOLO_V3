

def refine_box(boxes):
    output = []
    for box in boxes:
        x_center = (int(box[2]) - int(box[0])) / 2
        y_center = (int(box[3]) - int(box[1])) / 2
        w = int(box[3]) - int(box[1])
        h = int(box[2]) - int(box[0])
        box = [x_center, y_center, w, h]
        output.append(box)
    return output