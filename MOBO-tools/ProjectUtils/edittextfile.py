import shutil

def replace_line_in_file(file_path, line_number, new_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()  

    if 0 <= line_number < len(lines):
        lines[line_number] = new_line + '\n' 

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
    else:
        print(f"Invalid line number: {line_number}")
        

def create_dat(parameters, jobid):
    #create and edit variation dat file
    dat_init = str(os.environ["AIDE_HOME"])+"/variation_copy.dat"
    dat_job = str(os.environ["AIDE_HOME"])+"/rich/variations/dat/variation_{}.dat".format(jobid))
    shutil.copyfile(dat_init, dat_job)

    line_number_dat = 1  
    new_line_dat = '  4\t0\t0\tparameters['dx']\tparameters['dy']\tparameters['dz']\tparameters['dthx']\tparameters['dthy']\tparameters['dthz']'
    
    replace_line_in_file(dat_job, line_number_dat, new_line_dat)
    
    return

def create_yaml(jobid):
    #create and edit yaml file                                                                                                      
    yaml_init = str(os.environ["AIDE_HOME"])+"/rich_test.yaml"
    yaml_job = str(os.environ["AIDE_HOME"])+"/rich/variations/yaml/rich_{}.yaml".format(jobid))
    shutil.copyfile(yaml_init, yaml_job)

    line_number_yaml = 11
    new_line_yaml = str('      variation: variation_{}'.format(jobid))

    replace_line_in_file(yaml_job, line_number_yaml, new_line_yaml)

    return
