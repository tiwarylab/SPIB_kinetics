# hyperparameters

## Selected hyperparameters to train an SPIB model
dt = 1000
rc_dim = 2 
num_labels = 20
encoder_type = 'Linear' 
neuron_num1 =  32 
neuron_num2 =64
## heuristic for batch size is to set it ~ 0.1% of the trajectory length
batch_size = 128 
learning_rate = 0.00001
beta = 0.0001  
seed = 0 
SPIB_OPs = 28
Weighted_or_not = "Unweighted"



import numpy as np
import torch
import csv
from SPIB_scripts import SPIB
import torch.nn.functional as F
import os 

# os.chdir('SPIB_scripts')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu', 0)

path = "SPIB/" + Weighted_or_not + "_d=" + str(rc_dim) + "_t=" + str(dt) + "_b=" + "{:.4f}".format(beta) + "_learn=" + "{:.6f}".format(learning_rate)

data_shape=SPIB_OPs
# Update Label should be set to False for testing purposes only
UpdateLabel = True
# Load patameters from SPIB checkpoint
restore_path= path + 'cpt0/IB_final_cpt.pt'
representative_inputs_path=path + '_representative_inputs' + str(seed) + '.npy'
representative_inputs = torch.tensor(np.load(representative_inputs_path))
IB = SPIB.SPIB(encoder_type, rc_dim, num_labels, data_shape, device, UpdateLabel, neuron_num1, neuron_num2)
IB.reset_representative(representative_inputs)
checkpoint=torch.load(restore_path,map_location ='cpu')
IB.load_state_dict(checkpoint['state_dict'])

IB.to(device)

# os.chdir('../')
# %cd ../

concatenate_data_mean = np.load('data/FKBP_distance_data_mean.npy')
concatenate_data_std =  np.load('data/FKBP_distance_data_std.npy')
#Find the std from the unbiased MD file'COLVAR_unbiased_reduced'

# load the distance mean and std
distance_data_mean = np.load('data/FKBP_distance_data_mean.npy')
distance_data_std = np.load('data/FKBP_distance_data_std.npy')

# load data
distance_data = np.loadtxt("data/COLVAR_unbiased")
distance_data = distance_data[:,1:29]

normalized_traj_data = (distance_data-distance_data_mean)/(distance_data_std)
normalized_traj_data = torch.from_numpy(normalized_traj_data).float().to(device)

batch_size = 512 

# pass through VAE

all_z_mean=[]

for i in range(0, len(normalized_traj_data), batch_size):
    
    batch_inputs = normalized_traj_data[i:i+batch_size].to(device)

    # pass through VAE
    # log_prediction, z_sample, z_mean, z_logvar = self.forward(batch_inputs)
    z_mean, z_logvar = IB.encode(batch_inputs)
    
    all_z_mean+=[z_mean.cpu()]
    
all_z_mean = torch.cat(all_z_mean, dim=0).data.numpy()

np.save('FKBP_unbiased_mean_representation.npy', all_z_mean)

print("===== Second updated sigma for RC 1 and RC2 ======")
print(r'[sigma_1,sigma_2] = ',all_z_mean.std(axis=0))

# print(all_z_mean.std(axis=0))

# Using the std found using unbiased MD, and the RC obtained using SPIB generate plumed file 

sigma_1, sigma_2 = all_z_mean.std(axis=0)
sigma_1 = round(sigma_1,2)
sigma_2 = round(sigma_2,2)

height = 1.5 
bias_factor = 10 
pace = 5000
Nround=2

weight0=IB.encoder_mean.weight.cpu().data.numpy()
bias0=IB.encoder_mean.bias.cpu().data.numpy()

fi = open("data/plumed_header.txt",'r')

header = fi.readlines()

fi.close()

# for distances
CategoryI = list(range(26,108,3)) #to specify every 3th residue of the carbon atom in the protein 

CategoryII = ['r1'] # COM of each 5 rings in the imatinb 

# load the distance mean and std
distance_data_mean = np.load('data/FKBP_distance_data_mean.npy')
distance_data_std = np.load('data/FKBP_distance_data_std.npy')

with open("data/plumed_metaD_ANN.dat", 'w') as f:
    f.writelines(header)
    f.write('\n')
    
    output_OPs = ''
    
    # obtain the distances
    for i in CategoryI:
        for j in CategoryII:
            f.write("d_r%d_%s: DISTANCE ATOMS=@CA-%d,%s\n"%(i,j,i,j))
            output_OPs = output_OPs + "d_r%d_%s,"%(i,j)
    
    f.write('\n')
    
    output_OPs = output_OPs[:-1]
    f.write('\n')
    
    # normalize the OPs
    normalized_OPs = ''
    k = 0
    for i in CategoryI:
        for j in CategoryII:
            printfun = "(x-%0.12f)/(%0.12f)" % (distance_data_mean[k],distance_data_std[k])
            f.write("nd_r%d_%s: MATHEVAL ARG=d_r%d_%s FUNC=%s PERIODIC=NO\n" % (i,j,i,j,printfun))
            normalized_OPs = normalized_OPs + "nd_r%d_%s,"%(i,j)
            k+=1
            
    f.write('\n')
    normalized_OPs = normalized_OPs[:-1]
    f.write('\n')
    
    # 1st layer
    for i in range(weight0.shape[0]):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        toprint = toprint + normalized_OPs
                
        toprint = toprint + " COEFFICIENTS="
        for j in range(weight0.shape[1]):
            toprint = toprint + "%0.12f," % (weight0[i,j])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        f.write(toprint)
    for i in range(weight0.shape[0]):
        onebias = bias0[i]
        
        if onebias >= 0:
            printfun = "x+%0.12f" % (onebias)
        else:
            printfun = "x-%0.12f" % (-onebias)
        f.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
    
    output_variable=''
    for i in range(weight0.shape[0]):
        output_variable=output_variable+"l1r_%i," % (i+1)
    output_variable=output_variable[:-1]
    
    f.write('\n')
    f.write('# load metaD\n')
            
    f.write(f'\nMETAD ...\nLABEL=metad\nARG='+output_variable+f' SIGMA={sigma_1},{sigma_2} HEIGHT={height} BIASFACTOR={bias_factor} TEMP=300.0 PACE={pace}\nGRID_MIN=-30,-30 GRID_MAX=15,20 GRID_BIN=2500,3000\nCALC_RCT RCT_USTRIDE=1\n... METAD\n')
    f.write('\n')
    
    f.write('\n')
    f.write('# load metaD\n')
    f.write('\nCOMMITTOR ...\nARG=d1\nSTRIDE=10\nBASIN_LL1=3.0\nBASIN_UL1=10\n... COMMITTOR')
    
    f.write('\n')
    
    f.write('\nPRINT ARG='+output_OPs+',h_bond,d1,'+output_variable+',metad.bias,metad.rbias STRIDE=10 FILE=COLVAR_round_%i'%(Nround))
    f.write('\nDUMPMASSCHARGE FILE=mcfile')
    
    