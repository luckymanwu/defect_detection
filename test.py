import os.path
from shutil import copyfile

source_file = "C:\\Users\\Administrator\\Desktop\\cxData\\Annotations\\1064.xml"
destination = "C:\\Users\\Administrator\\Desktop\\cxData\\Annotations"
count = 1330
save_name = os.path.join(destination,str(count).zfill(4)+".xml")
copyfile(source_file, save_name)
