import os, sys, uproot, numpy as np, awkward as ak, xml.etree.ElementTree as ET
from sklearn.metrics import roc_auc_score

# lengths in mm
# superlayer geom:
# steel layer | air gap 1 | scintillating layer 1 | air gap 2 | air gap 3 | scintillating layer 2 | air gap 4

xmlfile = os.path.join(os.environ['EPIC_HOME'], f'compact/pid/klmws_{sys.argv[4]}.xml')
#xmlfile = os.path.join(os.environ['EPIC_HOME'], 'compact', 'pid', 'klmws.xml')
root = ET.parse(xmlfile).getroot()

superlayer_count = int(root.find(".//constant[@name='HcalScintillatorNbLayers']").get('value')) # number of superlayers
steel_thick = float(root.find(".//constant[@name='HcalSteelThickness']").get('value')[:-3]) # thickness of steel sublayer
sens_sublayer_thick = float(root.find(".//constant[@name='HcalScintillatorThickness']").get('value')[:-3]) # thickness of sensitive sublayer

inner_radius = 1770 # starting radial position of first layer
air_gap_thick = 0.3 # thickness of air gap between sublayers
first_sens_sublayer_pos = inner_radius + steel_thick + air_gap_thick  # position of first sensitive sublayer
superlayer_dist = steel_thick + sens_sublayer_thick * 2 + air_gap_thick * 4 # distance between superlayer starting points (i.e. width of superlayer)
adj_sens_sublayer_dist = sens_sublayer_thick + air_gap_thick * 2 # distance between adjacent sensitive sublayers within the same superlayer
outer_radius = inner_radius + superlayer_count * superlayer_dist # outer radius of barrel

sector_count = 8 # number of radial sectors
barrel_length = 1500 # length of the barrel along the z-axis
barrel_offset = 18 # offset of the barrel in the positive z direction

# array containing the start pos of each sensitive sublayer
layer_pos = np.zeros(superlayer_count * 2) # 2 sensitive sublayers per superlayer
layer_pos[::2] = [first_sens_sublayer_pos + superlayer_dist*i for i in range(superlayer_count)] # first sublayers 
layer_pos[1::2] = layer_pos[::2] + adj_sens_sublayer_dist # second sublayers



# returns the distance in the direction of the nearest sector
def sector_proj_dist(xpos, ypos):
    sector_angle = (np.arctan2(ypos, xpos) + np.pi / sector_count) // (2*np.pi / sector_count) * 2*np.pi / sector_count # polar angle (in radians) of the closest sector
    return xpos * np.cos(sector_angle) + ypos * np.sin(sector_angle) # scalar projection of position vector onto unit direction vector 
  
# returns the layer number for the position of a detector hit
def layer_num(xpos, ypos):
    pos = sector_proj_dist(xpos, ypos)
    
    # false if hit position is before the first sensitive sublayer or after the last sensitive sublayer 
    within_layer_region = np.logical_and(pos * 1.0001 > layer_pos[0], pos / 1.0001 < layer_pos[-1] + sens_sublayer_thick)

    superlayer_index = np.where(within_layer_region, ak.values_astype( (pos * 1.0001 - layer_pos[0]) // superlayer_dist, 'int64'), -1) # index of superlayer the hit may be in, returns -1 if out of region
    layer_pos_dup = ak.Array(np.broadcast_to(layer_pos, (int(ak.num(superlayer_index, axis=0)), len(layer_pos)))) # turn layer_pos into a 2d array with duplicate rows to allow indexing
    dis_from_first_sublayer = np.where(within_layer_region, pos - layer_pos_dup[superlayer_index * 2], -1) # distance of hit from the first sublayer in the superlayer, returns -1 if out of region
    
    # true if hit is within the first of the paired layers
    in_first_layer = np.logical_and(within_layer_region, dis_from_first_sublayer / 1.0001 <= sens_sublayer_thick)
    # true if hit is within the second of the paired layers
    in_second_layer = np.logical_and(within_layer_region, np.logical_and(dis_from_first_sublayer * 1.0001 >= adj_sens_sublayer_dist, dis_from_first_sublayer / 1.0001 <= adj_sens_sublayer_dist + sens_sublayer_thick))

    # layer number of detector hit; returns -1 if not in a layer
    hit_layer = np.where(in_first_layer, superlayer_index * 2 + 1, -1)
    hit_layer = np.where(in_second_layer, superlayer_index * 2 + 2, hit_layer)
    return hit_layer

# returns the number of pixels detected by a hit
def pixel_num(energy_dep, zpos):
    inverse = lambda x : 494.98 / (29.9733 - x + barrel_length / 2) - 0.16796
    efficiency = inverse(zpos - barrel_offset) + inverse(barrel_offset - zpos) # ratio of photons produced in a hit that make it to the sensor
    return 10 * energy_dep * (1000 * 1000) * efficiency

# takes in position and energy deposited for a hit as ragged 2d awkward arrays
# with each row corresponding to hits produced by a particle and its secondaries
# returns 1d numpy array containing number of layers traveled by each track (-1: hits not in any layer, -2: hits produce too few pixels, -3: no hits for this particle)
# and a 1d numpy array containing number of terminating tracks for each layer, starting at layer 1 
def layer_calc(xpos, ypos, zpos, energy_dep):
    hit_layer = layer_num(xpos, ypos)
    hit_layer_filtered = np.where(pixel_num(energy_dep, zpos) >= 2, hit_layer, -2) # only accept layers with at least 2 pixels
    layers_traveled = ak.fill_none(ak.max(hit_layer_filtered, axis=1), -3) # max accepted layers traveled for a track determines total layers traveled
    layer_counts = np.asarray(ak.sum(layers_traveled[:, None] == np.arange(1, superlayer_count * 2 + 1), axis=0)) # find counts for each layer, 1 through max
    return np.asarray(layers_traveled)[layers_traveled >= 1], layer_counts



with uproot.open(sys.argv[1]) as mu_file:
    mu_hit_x = mu_file['events/HcalBarrelHits.position.x'].array()
    mu_hit_y = mu_file['events/HcalBarrelHits.position.y'].array()
    mu_hit_z = mu_file['events/HcalBarrelHits.position.z'].array()
    mu_hit_edep = mu_file['events/HcalBarrelHits.EDep'].array()

with uproot.open(sys.argv[2]) as pi_file:
    pi_hit_x = pi_file['events/HcalBarrelHits.position.x'].array()
    pi_hit_y = pi_file['events/HcalBarrelHits.position.y'].array()
    pi_hit_z = pi_file['events/HcalBarrelHits.position.z'].array()
    pi_hit_edep = pi_file['events/HcalBarrelHits.EDep'].array()

mu_layers_traveled, mu_layer_counts = layer_calc(mu_hit_x, mu_hit_y, mu_hit_z, mu_hit_edep)
pi_layers_traveled, pi_layer_counts = layer_calc(pi_hit_x, pi_hit_y, pi_hit_z, pi_hit_edep)



layers_traveled_tot = np.concatenate((mu_layers_traveled, pi_layers_traveled))
# 1 if particle is muon, 0 if particle is pion
pid_actual = np.concatenate((np.ones_like(mu_layers_traveled), np.zeros_like(pi_layers_traveled)))
# probability that a particle stopping at this layer is a muon
pid_layer_prob = np.divide(mu_layer_counts, mu_layer_counts + pi_layer_counts, out=np.zeros(mu_layer_counts.size), where=(mu_layer_counts+pi_layer_counts)!=0)
# probability that each particle is a muon
pid_model = pid_layer_prob[layers_traveled_tot - 1]


# calculate ROC AUC score
roc_score = roc_auc_score(pid_actual, pid_model)

np.savetxt(sys.argv[3], (roc_score, outer_radius))


# load = np.loadtxt(sys.argv[3])
# print(f'AUC: {load[0]}')
# print(f'Outer radius: {load[1]}')