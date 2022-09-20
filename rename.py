import os
count = 1
path = "C:\\Users\\Administrator\\Desktop\\trash\\Annotations"
filelist = os.listdir(path)

for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    filetype = os.path.splitext(file)[1]
    Newdir = os.path.join(path, str(count).zfill(4) + filetype)
    os.rename(Olddir, Newdir)

    count += 1

