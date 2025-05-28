
import os
import xml.etree.ElementTree as ET
import sys
import json
import shutil
import math

def getPath(param, configfile):
    if(os.path.isfile(configfile) == False):
        print ("ERROR: parameter config file does not exist")
        sys.exit(1)
    with open(configfile) as f:
        params = json.loads(f.read())["parameters"]
        if params[param]:
            name = params[param]["element"]
            path = params[param]["path"]
            units = params[param]["units"]
            return path, name, units
        else:
            print("could not find parameter info")            
            return -1, -1, -1 

def editGeom(param, value, jobid, parameters):
    if jobid == -1:        
        xmlfile = str(os.environ['DETECTOR_PATH']+"/compact/pid/drich.xml")
    else:
        xmlfile = str(os.environ['DETECTOR_PATH']+"/compact/pid/drich_{}.xml".format(jobid))        
    
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    paramfile = str(os.environ['AIDE_HOME']+"/parameters.config")

    path, elementToEdit, units = getPath(param, paramfile)
    if path == -1:
        print("ERROR: element path not found/defined")
    element = root.find(path)
    current_val = element.get(elementToEdit)

    '''
    if ("radius" in param) and ("mirror" in param):
        # set center z based on mirror radius
        z_path = path.replace('radius','centerz')
        z_element = root.find(z_path)
        z_elementToEdit = elementToEdit.replace('radius','centerz')
        # fix position to drich backplane
        z_element.set(z_elementToEdit,"{}*{}".format(314 - value, units))
    else:
    '''
    if units != '':
        element.set(elementToEdit,"{}*{}".format(value,units))        
    else:
        element.set(elementToEdit,"{}".format(value))

    tree.write(xmlfile)
    return
    
def editEPIC(xml, jobid):
    # load drich_{jobid}.xml in the epic_craterlake_{jobid}.xml
    path = "${DETECTOR_PATH}/compact/pid/"
    drich_old = "drich.xml"
    drich_new = "drich_{}.xml".format(jobid)
    tree = ET.parse(xml)
    root = tree.getroot()

    for element in root.findall('.//include'):
        if element.get('ref') == str(path+drich_old):
            element.set('ref', str(path+drich_new))
            tree.write(xml)
            return
    print("failed to update to new drich geo")
    return
        
def create_xml(parameters, jobid):
    #create new epic xml    
    epic_xml = "{}/{}.xml".format(os.environ['DETECTOR_PATH'],os.environ['DETECTOR_CONFIG'])
    epic_xml_job = "{}/{}_{}.xml".format(os.environ['DETECTOR_PATH'],os.environ['DETECTOR_CONFIG'],jobid) 
    shutil.copyfile(epic_xml, epic_xml_job)
    #change drich.xml -> drich_{jobid}.xml
    editEPIC(epic_xml_job, jobid)
    
    #create and edit drich xml
    drich_xml = str(os.environ['DETECTOR_PATH']+"/compact/pid/drich.xml")
    drich_xml_job = str(os.environ['DETECTOR_PATH']+"/compact/pid/drich_{}.xml".format(jobid))
    shutil.copyfile(drich_xml, drich_xml_job)

    for param in parameters:        
        editGeom(param, parameters[param], jobid, parameters)    
    return

    
