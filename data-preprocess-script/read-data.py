def read_div(name):
    div = [] 
    index = -1
    with open(name, "r") as f:
        lines = f.readlines()
        for l in lines:
            index += 1
            print(index)
            dig = l.split()
            for d in dig:
                div.append(float(d))
    return div
