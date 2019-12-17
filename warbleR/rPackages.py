import subprocess

path = "D:/Work/2019-2020/IA/IA_WarbleR/" #Change this variable to the path to extrated project
arg = [path + "wavs"]
command = 'Rscript'

cmd1 = [command, path + 'packageInstalleR.R'] + arg #Install required R packages
cmd2 = [command, path + 'packageCheckeR.R'] + arg #Check WarbleR works


subprocess.check_call(cmd2, shell=False)

