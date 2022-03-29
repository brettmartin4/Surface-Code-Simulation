# File for formatting the data from the parallel surface code simulation, which usually ends up appending a bunch of extra null characters because of how the file view is set up in MPI

import csv

oldcsv = open('testfile.csv', 'rb')
data = oldcsv.read()
oldcsv.close()

newcsv = open('testfile_n.csv', 'wb')
newcsv.write(data.replace(b'\x00', b''))
newcsv.close()


