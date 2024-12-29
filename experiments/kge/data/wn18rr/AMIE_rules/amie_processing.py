with open('../train.txt') as f:
    lines = f.readlines()
    lines = [line.replace('\t', ' ') for line in lines]
file = open('train_amie_processed.txt', 'w+')
for line in lines:
    file.write(line)

print('end')
