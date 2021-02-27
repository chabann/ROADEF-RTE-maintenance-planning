from main_all import main_all
from main_cma_vns import main

t = 900 # time limit to stop the program execution after time limit seconds:
#              900 for 15 minutes and 5400 for 1,5 hour

p = 'A_02'    # instance name to load the data associated with the instance

o = 'output/' + p + ".txt"   # solution file name

name = 'J55'   # teamId that is the author of the executable

s = 100

#main_all(t, p, o, name, s)  # uncomment if you want to execute all algorithms

main(t, p, o, name, s)
