import sys

nc_file = sys.argv[1]
nc_adv_file = sys.argv[2]

file1 = open(nc_file, 'r')
lines = file1.readlines()
nc_cov = {}
for l in lines:
    l = l.strip()
    sp = l.split(' ')
    if len(l) < 3:
        continue
    nc_cov[(sp[0],sp[1])] = sp[2]
file1.close()

file2 = open(nc_adv_file, 'r')
lines = file2.readlines()
for l in lines:
    l = l.strip()
    sp = l.split(' ')
    if len(l) < 3:
        continue
    if sp[2] == "True":
        nc_cov[(sp[0],sp[1])] = "True"
file1.close()


cnt = 0.0
for m,l in nc_cov:
    if nc_cov[(m,l)] == "True":
        cnt = cnt + 1

print(100*cnt/len(nc_cov))

