
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

def editGeom(param, value, jobid,num_layers):
    
    extra_params = {
        "HcalScintillatorThickness":".//constant[@name='HcalScintillatorThickness']",
        "HcalSteelThickness" : ".//constant[@name='HcalSteelThickness']"
    }
    
    if jobid == -1:        
        xmlfile = str(os.environ['EPIC_HOME']+"/compact/pid/klmws.xml")
    else:
        xmlfile = str(os.environ['EPIC_HOME']+"/compact/pid/klmws_{}.xml".format(jobid))        

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    paramfile = str(os.environ['AIDE_HOME']+"/parameters.config")
    
    path, elementToEdit, units = getPath(param, paramfile)
    if path == -1:
        print("ERROR: element path not found/defined")
#     print(f"{param}")
    element = root.find(path)
#     current_val = element.get(elementToEdit)

    if param == 'num_layers':
        # num_layers only takes on integer values
        element.set(elementToEdit,"{}".format(int(value)))
    elif param == 'steel_ratio':
        total = (55.5 + 20 + 0.3 * 2) * 14
        units = "mm"
        total_per_layer = (total / num_layers) - (0.3 * 2)
        steel_value = total_per_layer * value
        scint_value = total_per_layer - steel_value
        
        scint_element = root.find(extra_params["HcalScintillatorThickness"])
        steel_element = root.find(extra_params["HcalSteelThickness"])
        
        scint_element.set("value","{}*{}".format(scint_value,units))        
        steel_element.set("value","{}*{}".format(steel_value,units))   
    elif units != '':     
        element.set(elementToEdit,"{}*{}".format(value,units))        
    else:
        element.set(elementToEdit,"{}".format(value))
    
    tree.write(xmlfile)
    return

def editEPIC(xml, jobid):
    path = "${DETECTOR_PATH}/compact/pid/"
    klmws_old = "klmws.xml"
    print(f"epic path: path")
    klmws_new = "klmws_{}.xml".format(jobid)
    tree = ET.parse(xml)
    root = tree.getroot()

    for element in root.findall('.//include'):
        if element.get('ref') == str(path+klmws_old):
            element.set('ref', str(path+klmws_new))
            tree.write(xml)
            return
    print("failed to update to new klmws geo")
    return
    
    
def create_xml(parameters, jobid):
    #create new epic xml
    epic_xml = "{}/{}.xml".format(os.environ['DETECTOR_PATH'],os.environ['DETECTOR_CONFIG'])
    epic_xml_job = "{}/{}_{}.xml".format(os.environ['DETECTOR_PATH'],os.environ['DETECTOR_CONFIG'],jobid) 
    shutil.copyfile(epic_xml, epic_xml_job)
    #change epic_klmws_only.xml -> epic_klmws_only_{jobid}.xml
    editEPIC(epic_xml_job, jobid)
    
    #create and edit klmws xml
    klmws_xml = str(os.environ['EPIC_HOME']+"/compact/pid/klmws.xml")
    klmws_xml_job = str(os.environ['EPIC_HOME']+"/compact/pid/klmws_{}.xml".format(jobid))
    shutil.copyfile(klmws_xml, klmws_xml_job)
    num_layers = parameters['num_layers']
    for param in parameters:
        editGeom(param, parameters[param], jobid,num_layers)
    return

    
