
from colorama import init
from test1 import test1
from test2 import test2

init()

def runTest():
    print('----------')
    print('TEST BEGIN')
    print('----------')

    test1()
    test2()

    print('----------')
    print('TEST END')
    print('----------')

runTest()
