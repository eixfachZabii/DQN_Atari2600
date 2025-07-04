import pyautogui

import time

import random

while True:
    x = random.randint(0,1000)

    y = random.randint(0,1000)

    pyautogui.moveTo(x,y)

    localtime = time.localtime()

    result = time.strftime("%I:&M:%S %p", localtime)

    print('Moved at ' + str(result) + ' ('  + str(x) + ', ' + str(y) + ')')

    time.sleep(10)