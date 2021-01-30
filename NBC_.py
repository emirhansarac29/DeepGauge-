import sys

nbc_file = sys.argv[1]
nbc_adv_file = sys.argv[2]

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

file1 = open(nbc_adv_file, 'r')
lines = file1.readlines()
nbc_adv_cov = {}
for l in lines:
    l = l.strip()
    sp = l.split(' ')
    if len(l) < 3:
        continue
    nbc_adv_cov[(sp[0], sp[1])] = [float(sp[2]), float(sp[3])]
file1.close()

last_cov = {}

for k,l in nbc_cov:
    a1 = nbc_cov[(k,l)]
    a2 = nbc_adv_cov[(k, l)]
    t = 0
    for i in range(len(a1)):
        if (a1[i] == 1) or (a2[i] == 1):
            t = t + 1
    last_cov[(k,l)] = t


cnt = 0.0

for k,l in last_cov:
    cnt = cnt + last_cov[k,l]

print(100*cnt/(len(nbc_cov)*2))