import sys

kmnc_file = sys.argv[1]
kmnc_adv_file = sys.argv[2]

file1 = open(kmnc_file, 'r')
lines = file1.readlines()

kmnc_cov = {}
for l in lines:
    l = l.strip()
    sp = l.split(' ')
    if len(l) < 3:
        continue
    v = []
    for t in range(2,len(sp)):
        v.append(float(sp[t]))
    kmnc_cov[(sp[0], sp[1])] = v
file1.close()

file1 = open(kmnc_adv_file, 'r')
lines = file1.readlines()

kmnc_adv_cov = {}
for l in lines:
    l = l.strip()
    sp = l.split(' ')
    if len(l) < 3:
        continue
    #kmnc_cov[(sp[0], sp[1])]
    v = []
    for t in range(2,len(sp)):
        v.append(float(sp[t]))
    kmnc_adv_cov[(sp[0], sp[1])] = v
file1.close()

last_cov = {}

for k,l in kmnc_cov:
    a1 = kmnc_cov[(k,l)]
    a2 = kmnc_adv_cov[(k, l)]
    t = 0
    for i in range(len(a1)):
        if (a1[i] == 1) or (a2[i] == 1):
            t = t + 1
    last_cov[(k,l)] = t


div = 0.0
cnt = 0.0

for k,l in last_cov:
    cnt = cnt + last_cov[k,l]
    div = div + len(kmnc_cov[k,l])

print(100*cnt/div)