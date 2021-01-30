import sys

tknp_file = sys.argv[1]
tknp_adv_file = sys.argv[2]

tknp_cov = set()

file1 = open(tknp_file, 'r')
lines = file1.readlines()
for l in lines:
    l = l.strip()
    if l == "":
        continue
    tknp_cov.add(l)
file1.close()

file1 = open(tknp_adv_file, 'r')
lines = file1.readlines()
for l in lines:
    l = l.strip()
    if l == "":
        continue
    tknp_cov.add(l)
file1.close()

print(len(tknp_cov))

