
import os
import xml.etree.ElementTree as ET
import sys

def getPath(param):
    if param == "focus_tune_x" or param == "focus_tune_z":
        path = ".//detector/mirror/[@{}]".format(param)
        element = param
    else:
        return -1, -1
    return path, element
    
def editGeom(param, value):    
    xmlfile = str(os.environ['EPIC_HOME']+"/compact/pid/drich.xml")
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    path, elementToEdit = getPath(param)
    element = root.find(path)
    current_val = element.get(elementToEdit)
    # hardcode units (has to be a better way)
    if param == "focus_tune_x" or param == "focus_tune_z":
        units = "cm"    
    if units != '':
        element.set(elementToEdit,"{}".format(value))        
    else:
        element.set(elementToEdit,"{}*{}".format(value,units))
    tree.write(xmlfile)
    return

