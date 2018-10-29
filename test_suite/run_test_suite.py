from TEST1_variable_EDOS_variable_gap_1d import runTest1
from TEST2_variable_epsilon_1d import runTest2
from TEST3_singleGB_homojunction_2d_periodic import runTest3
from TEST4_singleGB_homojunction_2d_abrupt import runTest4
from TEST5_variable_epsilon_2d_abrupt import runTest5
from TEST6_variable_epsilon_2d_periodic import runTest6
from TEST7_variable_gap_2d_pillars_abrupt import runTest7
from TEST8_variable_gap_2d_pillars_periodic import runTest8


print("\nrunning test suite: error should be less than 0.01 for all sims")
print("\n----------------------------------------------------------------")

print("\nrunning test 1: 1d variable electronic structure")
runTest1()

print("\nrunning test 2: 1d variable epsilon")
runTest2()

print("\nrunning test 3: 2d single GB periodic b.c.")
runTest3()

print("\nrunning test 4: 2d single GB abrupt b.c.")
runTest4()

print("\nrunning test 5: 2d variable epsilon abrupt b.c.")
runTest5()

print("\nrunning test 6: 2d variable epsilon periodic b.c.")
runTest6()

print("\nrunning test 7: 2d variable electronic structure abrupt b.c.")
runTest7()

print("\nrunning test 8: 2d variable electronic structure periodic b.c.")
runTest8()
