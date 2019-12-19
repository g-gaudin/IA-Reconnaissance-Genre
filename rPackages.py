import subprocess

command = 'Rscript'

cmd1 = [command, 'scriptsR/packageInstalleR.R']
cmd2 = [command, 'scriptsR/packageCheckeR.R']


subprocess.check_call(cmd2, shell=False)

