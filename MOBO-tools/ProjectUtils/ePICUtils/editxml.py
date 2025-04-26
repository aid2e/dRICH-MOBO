
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

def editGeom(param, value, jobid,num_layers,preshower_ratio):
    
    extra_params = {
        "HcalScintillatorThickness":".//constant[@name='HcalScintillatorThickness']",
        "HcalSteelThickness" : ".//constant[@name='HcalSteelThickness']",
        "preshower_scint_value" : ".//constant[@name='preshower_scint_value']",
        "preshower_steel_value" : ".//constant[@name='preshower_steel_value']",
        "postshower_scint_value" : ".//constant[@name='postshower_scint_value']",
        "postshower_steel_value" : ".//constant[@name='postshower_steel_value']"
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
    steel_element = root.find(extra_params["HcalSteelThickness"])
    scint_element = root.find(extra_params["HcalScintillatorThickness"])
    steel_value = float(steel_element.get('value')[:-3])
    scint_value = float(scint_element.get('value')[:-3])

    if param == 'num_layers':
        # num_layers only takes on integer values
        element.set(elementToEdit,"{}".format(int(value)))
    elif param == 'preshower_steel_ratio':
        total = (55.5 + 20 + 0.3 * 2) * 14
        units = "mm"
        total_steel_scint = total - (0.3 * 2) * num_layers
        
        total_steel_scint = (steel_value + scint_value) * num_layers
        pre_shower_steel_scint_per_layer = (total_steel_scint * preshower_ratio) / 2
        preshower_steel_value = pre_shower_steel_scint_per_layer * value
        preshower_scint_value = pre_shower_steel_scint_per_layer - preshower_steel_value
        
        preshower_scint_element = root.find(extra_params["preshower_scint_value"])
        preshower_steel_element = root.find(extra_params["preshower_steel_value"])
        
        preshower_scint_element.set("value","{}*{}".format(preshower_scint_value,units))        
        preshower_steel_element.set("value","{}*{}".format(preshower_steel_value,units))   
    elif param == 'postshower_steel_ratio':
        total = (55.5 + 20 + 0.3 * 2) * 14
        units = "mm"
        total_steel_scint = total - (0.3 * 2) * num_layers
        
        total_steel_scint = (steel_value + scint_value) * num_layers
        post_shower_steel_scint_per_layer = (total_steel_scint * (1 - preshower_ratio)) / (num_layers - 2)
        postshower_steel_value = post_shower_steel_scint_per_layer * value
        postshower_scint_value = post_shower_steel_scint_per_layer - postshower_steel_value
        
        postshower_scint_element = root.find(extra_params["postshower_scint_value"])
        postshower_steel_element = root.find(extra_params["postshower_steel_value"])
        
        postshower_scint_element.set("value","{}*{}".format(postshower_scint_value,units))        
        postshower_steel_element.set("value","{}*{}".format(postshower_steel_value,units))  
    elif units != '':     
        element.set(elementToEdit,"{}*{}".format(value,units))     
    elif param == 'preshower_ratio':
        return
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
    
    num_layers = parameters['num_layers'] if('num_layers' in parameters) else 14
    preshower_ratio = parameters['preshower_ratio'] if('preshower_ratio' in parameters) else (2.0/14.0)
    for param in parameters:
        editGeom(param, parameters[param], jobid,num_layers,preshower_ratio)
    return

    
