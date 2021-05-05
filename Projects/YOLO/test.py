import os
from YOLO_wrapper import *


if __name__ == '__main__':
    l = Label(("person", 0.91, (655.6371459960938, 381.6572570800781, 902.2772216796875, 654.2041015625)))
    print("person" in [l])