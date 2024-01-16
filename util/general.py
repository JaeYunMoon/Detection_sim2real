import os 
import shutil
from pathlib import Path
from enum import Enum

import cv2

def check_dir(directorys,nameArg):
    assert os.path.isdir(directorys),f"argument {nameArg} : Directory does not exist{directorys}"
    dirs = directorys.replace("\\",os.sep)
    files_list = os.listdir(dirs)
    f = [os.path.join(directorys,file) for file in files_list if file.endswith(".txt")]
    assert not len(f) == 0,f"argument {nameArg} : .txt files does not exist in Directory{directorys}"
    
    return f

def check_save_dir(directorys):
    if os.path.exists(directorys) and os.listdir(directorys):
        key_pressed = ''
        while key_pressed.upper() not in ["Y","N"]:
            print(f'Folder {directorys} already exists and may contain important results.\n')
            print(f'Enter \'Y\' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!')
            print(f'Or enter \'N\' to Create a new folder with a number.')
            key_pressed = input('')

        if str(key_pressed) == "Y":
            # shutil.rmtree(directorys, ignore_errors=True)
            #os.makedirs(directorys)
            print(f"Save Path : {directorys}")
            return directorys
        elif str(key_pressed)== "N":
            directorys = increment_path(directorys,mkdir=True)
            print(f"Save Path : {directorys}")
            return directorys
    elif not os.listdir(directorys):
        pass
    else:
        os.makedirs(directorys)
    return directorys
        
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}_{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def ValidateFormats(argFormat, argName):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        raise ValueError(f"{argName} : invalid value. It must be either \'xywh\' or \'xyrb\'")

class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    XYWH = 1
    XYX2Y2 = 2

class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    GroundTruth = 1
    Detected = 2

class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EveryPointInterpolation = 1
    ElevenPointInterpolation = 2