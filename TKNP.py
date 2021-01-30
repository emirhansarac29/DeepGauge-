import sys

tknp_file = sys.argv[1]

file1 = open(tknp_file, 'r')
lines = file1.readlines()

tknp_cov = set()
for l in lines:
    l = l.strip()
    if l == "":
        continue
    tknp_cov.add(l)
file1.close()

print(len(tknp_cov))

