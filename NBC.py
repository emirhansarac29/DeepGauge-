import sys

nbc_file = sys.argv[1]

file1 = open(nbc_file, 'r')
lines = file1.readlines()

nbc_cov = {}
for l in lines:
    l = l.strip()
    sp = l.split(' ')
    if len(l) < 3:
        continue
    nbc_cov[(sp[0], sp[1])] = [float(sp[2]), float(sp[3])]
file1.close()


cnt = 0.0

for k,l in nbc_cov:
    cnt = cnt + sum(nbc_cov[k,l])

print(100*cnt/(len(nbc_cov)*2))