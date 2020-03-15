import numpy as np
import random
import copy
import os


def read_txt_as_data(filename):
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))
    returnArray = np.array(returnArray)
    return returnArray



def define_actions(action):
    actions = ["walking","running","directing_traffic","soccer",
               "basketball","washwindow","jumping","basketball_signal"]
    if action in actions:
        return [action]
    elif action == "all":
        return actions
    else:
        raise( ValueError, "Unrecognized action: %d" % action )



def load_data(data_path, actions):
    nactions = len(actions)
    sampled_data_set, complete_data = {}, []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path='{}/{}'.format(data_path, action)
        count=0
        for fn in os.listdir(path):
            count=count+1
        for examp_index in np.arange(count):
            filename = '{}/{}/{}_{}.txt'.format(data_path, action, action, examp_index+1)
            action_sequence = read_txt_as_data(filename)
            t, d = action_sequence.shape
            even_indices = range(0, t, 2)
            sampled_data_set[(action, examp_index+1, 'even')] = action_sequence[even_indices, :]
            if len(complete_data) == 0:
                complete_data = copy.deepcopy(action_sequence)
            else:
                complete_data = np.append(complete_data, action_sequence, axis=0)
    return sampled_data_set, complete_data



def normalization_stats(complete_data):
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)
    dimensions_is_zero = []
    dimensions_is_zero.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_nonzero = []
    dimensions_nonzero.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_is_zero] = 1.0
    

    dim_to_ignore = [0,  1,  2,  3,  4,  5,  6,   7,   8,   21,  22,  23,  24,  25,  26, 
                     39, 40, 41, 60, 61, 62, 63,  64,  65,  81,  82,  83,
                     87, 88, 89, 90, 91, 92, 108, 109, 110, 114, 115, 116]
    dim_to_use = [9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31,  32,  33,  34,  35,  36,  37,  38, 
                  42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,  59,  66,  67,  68,  69,  70,  71,  72,  73,  74, 
                  75, 76, 77, 78, 79, 80, 84, 85, 86, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 111, 112, 113]
    return data_mean, data_std, dim_to_ignore, dim_to_use, dimensions_is_zero, dimensions_nonzero



def normalize_data(data_set, data_mean, data_std, dim_to_use):
    data_out = {}
    for key in data_set.keys():
        data_out[key] = np.divide((data_set[key]-data_mean), data_std)
        data_out[key] = data_out[key][:,dim_to_use]
    return data_out



def train_sample(data_set, batch_size, source_seq_len, target_seq_len, input_size):

    all_keys = list(data_set.keys())
    chosen_keys_idx = np.random.choice(len(all_keys), batch_size)
    total_seq_len = source_seq_len + target_seq_len

    encoder_inputs  = np.zeros((batch_size, source_seq_len-1, input_size), dtype=np.float32)
    decoder_inputs  = np.zeros((batch_size, 1, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=np.float32)

    for i in range(batch_size):
        the_key = all_keys[chosen_keys_idx[i]]
        t, d = data_set[the_key].shape
        idx = np.random.randint(0, t-total_seq_len)
        data_sel = data_set[the_key][idx:idx+total_seq_len,:]

        encoder_inputs[i,:,:]  = data_sel[0:source_seq_len-1,:]
        decoder_inputs[i,:,:]  = data_sel[source_seq_len-1:source_seq_len,:]
        decoder_outputs[i,:,:] = data_sel[source_seq_len:,:]

    return encoder_inputs, decoder_inputs, decoder_outputs



def find_indices_srnn(data_set, action):
    seed = 1234567890
    rng = np.random.RandomState(seed)
    subject = 5
    T1 = data_set[(subject, action, 1, 'even')].shape[0]
    T2 = data_set[(subject, action, 2, 'even')].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    return idx



def srnn_sample(data_set, action, source_seq_len, target_seq_len, input_size):
    batch_size = 8
    total_frames = source_seq_len + target_seq_len

    encoder_inputs = np.zeros((batch_size, source_seq_len-1, input_size), dtype=np.float32)
    decoder_inputs = np.zeros((batch_size, 1, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=np.float32)
    SEED = 1234567890
    rng = np.random.RandomState(SEED)
    for i in range(batch_size):
        data_sel = data_set[(action, 1, 'even')]
        t, _ = data_sel.shape
        idx = rng.randint(0, t-total_frames)
        data_sel = data_sel[idx :(idx + total_frames), :]
        encoder_inputs[i, :, :] = data_sel[0:source_seq_len-1, :]
        decoder_inputs[i, :, :] = data_sel[source_seq_len-1:source_seq_len, :]
        decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]

    return encoder_inputs, decoder_inputs, decoder_outputs



def expmap2rotmat(r):
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta+np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
    r0x = r0x-r0x.T
    R = np.eye(3,3)+np.sin(theta)*r0x+(1-np.cos(theta))*(r0x).dot(r0x)
    return R



def rotmat2euler(R):
    if R[0,2]==1 or R[0,2]==-1:
        e3 = 0
        dlta = np.arctan2(R[0,1], R[0,2])
        if R[0,2]==-1:
            e2 = np.pi/2
            e1 = e3+dlta
        else:
            e2 = -1*np.pi/2
            e1 = -1*e3+dlta
    else:
        e2 = -1*np.arcsin(R[0,2])
        e1 = np.arctan2(R[1,2]/np.cos(e2), R[2,2]/np.cos(e2))
        e3 = np.arctan2(R[0,1]/np.cos(e2), R[0,0]/np.cos(e2))
    eul = np.array([e1, e2, e3])
    return eul



def unnormalize_data(data, data_mean, data_std, dim_ignore, dim_use, dim_zero):
    t, d = data.shape[0], data_mean.shape[0]   # t = 25, d = 99
    orig_data = np.zeros((t, d), dtype=np.float32)    # [25, 99]
    mask = np.ones((t, d), dtype=np.float32)
    dim_use, dim_zero = np.array(dim_use), np.array(dim_zero)                       #  66
    orig_data[:,dim_use] = data
    orig_data[:,dim_zero] = 0.

    std_mat = np.repeat(data_std.reshape((1,d)), t, axis=0)
    mean_mat = np.repeat(data_mean.reshape((1,d)), t, axis=0)
    orig_data = np.multiply(orig_data, std_mat)+mean_mat
    return orig_data



def get_srnn_gts(actions, data_set, data_mean, data_std, dim_to_ignore, 
                 source_seq_len, target_seq_len, input_size, to_euler=True):
    srnn_gts = {}
    for action in actions:
        srnn_gt = []
        encoder_inputs, decoder_inputs, targets = srnn_sample(data_set, action, source_seq_len, target_seq_len, input_size)
        for i in np.arange(targets.shape[0]):
            target = targets[i,:,:]
            if to_euler:
                for j in np.arange(target.shape[0]):
                    for k in np.arange(0, 115, 3):
                        target[j,k:k+3] = rotmat2euler(expmap2rotmat(target[j,k:k+3]))
            srnn_gt.append(target)
        srnn_gts[action] = srnn_gt
    return srnn_gts


def quat2expmap(q):
    if (np.abs(np.linalg.norm(q)-1)>1e-3):
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta, coshalftheta = np.linalg.norm(q[1:]), q[0]
    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2*np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta+2*np.pi, 2*np.pi)

    if theta > np.pi:
        theta =  2*np.pi-theta
        r0 = -r0
    r = r0 * theta
    return r


def rotmat2quat(R):
    rotdiff = R-R.T;
    r = np.zeros(3)
    r[0] = -rotdiff[1,2]
    r[1] =  rotdiff[0,2]
    r[2] = -rotdiff[0,1]
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)
    sintheta = np.linalg.norm(r)/2
    costheta = (np.trace(R)-1)/2;
    theta = np.arctan2(sintheta, costheta);

    q = np.zeros(4)
    q[0] = np.cos(theta/2)
    q[1:] = r0*np.sin(theta/2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def _some_variables():
    parent = np.array([0, 1,  2,  3,  4,  5,  6,  1,  8,  9,  10, 11, 12, 1,  14, 15, 16, 17, 18,
                      19, 16, 21, 22, 23, 24, 25, 26, 24, 28, 16, 30, 31, 32, 33, 34, 35, 33, 37])-1
    offset = np.array([0,                0,                 0, 
                       0,                0,                 0,  
                       1.65674000000000, -1.80282000000000, 0.62477000000000,
                       2.59720000000000, -7.13576000000000, 0,  
                       2.49236000000000, -6.84770000000000, 0, 
                       0.19704000000000, -0.54136000000000, 2.14581000000000,
                       0,                0,                 1.11249000000000, 
                       0,                0,                 0,
                       -1.6107000000000, -1.80282000000000, 0.62476000000000,
                       -2.5950200000000, -7.12977000000000, 0,
                       -2.4678000000000, -6.78024000000000, 0,
                       -0.2302400000000, -0.63258000000000, 2.13368000000000,
                       0,                0,                 1.11569000000000,
                       0,                0,                 0,
                       0.01961000000000, 2.054500000000000, -0.1411200000000,
                       0.01021000000000, 2.064360000000000, -0.0592100000000,
                       0,                0,                 0, 
                       0.00713000000000, 1.567110000000000, 0.14968000000000, 
                       0.03429000000000, 1.560410000000000, -0.1000600000000,
                       0.01305000000000, 1.625600000000000, -0.0526500000000, 
                       0,                0,                 0, 
                       3.54205000000000, 0.904360000000000, -0.1736400000000,
                       4.86513000000000, 0,                 0,
                       3.35554000000000, 0,                 0,
                       0,                0,                 0, 
                       0.66117000000000, 0,                 0,
                       0.53306000000000, 0,                 0,
                       0,                0,                 0, 
                       0.54120000000000, 0,                 0.54120000000000,
                       0,                0,                 0,
                       -3.4980200000000, 0.759940000000000, -0.3261600000000, 
                       -5.0264900000000, 0,                 0,
                       -3.3643100000000, 0,                 0,
                       0,                0,                 0, 
                       -0.7304100000000, 0,                 0,
                       -0.5888700000000, 0,                 0,
                       0,                0,                 0,
                       -0.5978600000000, 0,                 0.59786000000000]).reshape(-1,3)
    
    rotInd = [[6,  5,  4],  [9,  8,  7],  [12, 11, 10], [15, 14, 13], [18, 17, 16], [21, 20, 19], [],
              [24, 23, 22], [27, 26, 25], [30, 29, 28], [33, 32, 31], [36, 35, 34], [],
              [39, 38, 37], [42, 41, 40], [45, 44, 43], [48, 47, 46], [51, 50, 49], [54, 53, 52], [],
              [57, 56, 55], [60, 59, 58], [63, 62, 61], [66, 65, 64], [69, 68, 67], [72, 71, 70], [],
              [75, 74, 73], [], 
              [78, 77, 76], [81, 80, 79], [84, 83, 82], [87, 86, 85], [90, 89, 88], [93, 92, 91], [],
              [96, 95, 94], []]

    posInd=[]
    for ii in np.arange(38):
        if ii==0:
            posInd.append([1,2,3])
        else:
            posInd.append([])

    expmapInd = np.split(np.arange(4,118)-1,38)
    return parent, offset, posInd, expmapInd


def revert_coordinate_space(channels, R0, T0):
    n, d = channels.shape
    channels_rec = copy.copy(channels)
    R_prev, T_prev = R0, T0
    rootRotInd = np.arange(3,6)

    for ii in range(n):
        R_diff = expmap2rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)
        channels_rec[ii, rootRotInd] = rotmat2expmap(R)
        T = T_prev + ((R_prev.T).dot(np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
        channels_rec[ii,:3] = T
        T_prev, R_prev = T, R
    return channels_rec


def fkl(angles, parent, offset, rotInd, expmapInd):
    njoints   = 38
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange( njoints ):
        try:
            if not posInd[i] : # If the list is empty
                xangle, yangle, zangle = 0, 0, 0
            else:
                xangle = angles[ posInd[i][2]-1 ]
                yangle = angles[ posInd[i][1]-1 ]
                zangle = angles[ posInd[i][0]-1 ]
        except:
            print (i)

        r = angles[ expmapInd[i] ]
        thisRotation = expmap2rotmat(r)
        thisPosition = np.array([xangle, yangle, zangle])
        if parent[i] == -1: # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array( xyz ).squeeze()
    xyz = xyz[:,[0,2,1]]

    return np.reshape( xyz, [-1] )


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot