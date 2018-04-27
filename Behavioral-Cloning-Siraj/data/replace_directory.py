import os
import sys
import fileinput

cwd = os.getcwd()  # Get the current working directory (cwd)
  # Get all the files in that directory


#print ("Text to search for:")
confirmation = input("IMG & csv files need to be in this directory , click yes if sure  ")

# enter the old directory of data (the whole direc before /IMG), 
textToSearch = "/home/user/Desktop/TrainingAllV2" 

# print ("enter:")
textToReplace = cwd

# print ("File to perform Search-Replace on:")
fileToSearch  = "driving_log.csv"
#fileToSearch = 'D:\dummy1.txt'

tempFile = open( fileToSearch, 'r+' )

for line in fileinput.input( fileToSearch ):
    tempFile.write( line.replace( textToSearch, textToReplace ) )
tempFile.close()

print("your csv file is modified with the new directory")
'''  if textToSearch in line :
        #print('Match Found')
        dummy = 0
    else:
        print('Match Not Found!!')
'''
# input( '\n\n Press Enter to exit...' )
