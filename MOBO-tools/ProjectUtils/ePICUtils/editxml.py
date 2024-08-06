
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
        params = json.loads(f.read())["parameters"]
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
    
    if param == "sensor_centerz":
        element.set(elementToEdit,"{}*{} - DRICH_zmin + 238*cm".format(value,units))
    elif param == "mirror_xcut":
        element.set(elementToEdit,"DRICH_rmax2/2 - {}*{}".format(value,units))
    else:
        if units != '':
            element.set(elementToEdit,"{}*{}".format(value,units))        
        else:
            element.set(elementToEdit,"{}".format(value))

        if ("radius" in param) and ("mirror" in param):
            # set center z based on mirror radius
            z_path = path.replace('radius','centerz')
            z_element = root.find(z_path)
            z_elementToEdit = elementToEdit.replace('radius','centerz')
            # fix position to drich backplane
            z_element.set(z_elementToEdit,"{}*{}".format(311 - value, units))
    tree.write(xmlfile)
    return

def editMirrorGeom(parameters, jobid):
    if jobid == -1:        
        xmlfile = str(os.environ['DETECTOR_PATH']+"/compact/pid/drich.xml")
    else:
        xmlfile = str(os.environ['DETECTOR_PATH']+"/compact/pid/drich_{}.xml".format(jobid))        

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    paramfile = str(os.environ['AIDE_HOME']+"/parameters.config")
    with open(paramfile) as f:
        paramconfig = json.loads(f.read())["parameters"]

        nmirrors = 3
        for i in range(1,nmirrors+1):
            # first update radius
            
            path, elementToEdit, units = getPath("mirror{}_radius".format(i), paramfile)
            element = root.find(path)
            radius = parameters["mirror{}_radius".format(i)]
            element.set(elementToEdit,"{}*{}".format(radius,"cm"))
            
            x_default = paramconfig["mirror{}_centerx".format(i)]["default"]
            x_shifted = x_default + (radius*parameters["mirror{}_centerx".format(i)])
            
            path, elementToEdit, units = getPath("mirror{}_centerx".format(i), paramfile)
            element = root.find(path)
            element.set(elementToEdit,"{}*{}".format(x_shifted,"cm"))

    tree.write(xmlfile)
    return
        
    
def editEPIC(xml, jobid):
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
    print("creating xmls ", jobid)
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
        if param == "sensor_centerz":
            editGeom(param, parameters[param] - parameters["sensor_radius"], jobid)
        elif "mirror" in param:
            continue
        else:
            editGeom(param, parameters[param], jobid)
    editMirrorGeom(parameters,jobid)
    return

    
