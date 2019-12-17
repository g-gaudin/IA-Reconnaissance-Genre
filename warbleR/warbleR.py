import csv
import subprocess

path = 'D:/Work/2019-2020/IA/IA_WarbleR/' #Change this variable to the path to extrated project
arg = [path + "wavs"]
command = 'Rscript'

cmd = [command, path + 'voiceAnalyzer.R'] + arg
subprocess.check_call(cmd, shell=False) #Run WarbleR script

#Now checking all extracted parameters
table = []
with open(path + 'wavs/results.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'{", ".join(row[3:15])}'+', '+f'{", ".join(row[16:])}')
            line_count += 1
        else:
            col_count = 0
            line = []
            for col in row:
                if col_count > 2 and col_count != 15:
                    line.append(float(col))
                col_count += 1
            table.append(line)
            line_count += 1

for row in table:
    print(row)
