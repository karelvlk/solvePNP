
from colorama import Fore

from calculateData import CALCULATE_DATA
from callService import callServiceCheck



def test1():
    index = 1
    print("Test calculating")
    for item in CALCULATE_DATA:
        print('INDEX', index)
        index += 1
        if (callServiceCheck(item['input'], item['output'])):
            print(Fore.GREEN + '  OK')
        else:
            print(Fore.RED + '  FAILED')
