ini_dict = {}
ptc_dict = {}
with open('train_id.txt','r') as file:
    for l in file.readlines():
        lines = l.strip('\n').split('\t')
        user = lines[0]
        item = lines[1]
        ptcs = lines[2:]
        if len(lines) == 2: continue
        if user in ini_dict:
            ini_dict[user].append(item)
        else:
            ini_dict[user] = [item]
        for user in ptcs:
            if user in ptc_dict:
                ptc_dict[user].append(item)
            else:
                ptc_dict[user] = [item]


file1 = open('train_ini.txt','w')
file2 = open('train_ptc.txt','w')


for user in ini_dict:
    line = user
    items = ini_dict[user]
    for item in items:
        line +=' '
        line += item
    line+='\n'
    file1.write(line)
for user in ptc_dict:
    line = user
    items = ptc_dict[user]
    for item in items:
        line +=' '
        line += item
    line+='\n'
    file2.write(line)
