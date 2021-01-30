import sys

kmnc_file = sys.argv[1]

file1 = open(kmnc_file, 'r')
lines = file1.readlines()

kmnc_cov = {}
for l in lines:
    l = l.strip()
    sp = l.split(' ')
    if len(l) < 3:
        continue
    #kmnc_cov[(sp[0], sp[1])]
    v = []
    for t in range(2,len(sp)):
        v.append(float(sp[t]))
    kmnc_cov[(sp[0], sp[1])] = v
file1.close()


div = 0.0
cnt = 0.0

for k,l in kmnc_cov:
    cnt = cnt + sum(kmnc_cov[k,l])
    div = div + len(kmnc_cov[k,l])

print(100*cnt/div)