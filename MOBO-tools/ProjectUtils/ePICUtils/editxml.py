
import os
import xml.etree.ElementTree as ET
import sys
import json
import shutil

def getPath(param, configfile):
    if(os.path.isfile(configfile) == False):
        print ("ERROR: parameter config file does not exist")
        sys.exit(1)
    with open(configfile) as f:
        params = json.loads(f.read())
        if params[param]:
            name = params[param]["element"]
            path = params[param]["path"]
            units = params[param]["units"]
            return path, name, units
        else:
            print("could not find parameter info")            
            return -1, -1, -1 

def editGeom(param, value, jobid):
    if jobid == -1:        
        xmlfile = str(os.environ['EPIC_HOME']+"/compact/pid/drich.xml")
    else:
        xmlfile = str(os.environ['EPIC_HOME']+"/compact/pid/drich_{}.xml".format(jobid))        

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    paramfile = str(os.environ['AIDE_HOME']+"/parameters.config")
    
    path, elementToEdit, units = getPath(param, paramfile)
    if path == -1:
        print("ERROR: element path not found/defined")
    element = root.find(path)
    current_val = element.get(elementToEdit)

    if elementToEdit == "centerz":
        element.set(elementToEdit,"{}*{} - DRICH_zmin".format(value,units))
    else:
        if units != '':
            element.set(elementToEdit,"{}*{}".format(value,units))        
        else:
            element.set(elementToEdit,"{}".format(value))
    tree.write(xmlfile)
    return

def editEPIC(xml, jobid):
    drich_old = "${DETECTOR_PATH}/compact/pid/drich.xml"
    drich_new = "${DETECTOR_PATH}/compact/pid/drich_{}.xml".format(jobid)
    tree = ET.parse(xml)
    root = tree.getroot()

    for element in root.findall('.//include'):
        if element.get('ref') == drich_old:
            include_element.set('ref', new_ref)
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
    drich_xml = str(os.environ['EPIC_HOME']+"/compact/pid/drich.xml")
    drich_xml_job = str(os.environ['EPIC_HOME']+"/compact/pid/drich_{}.xml".format(jobid))
    shutil.copyfile(drich_xml, drich_xml_job)

    for param in parameters:
        editGeom(param, parameters[param], jobid)
    return

    
