import yaml

def div_ceil(numerator, denominator):
    return (int)(numerator + denominator - 1) // denominator 

def get_weight_size(WPrecision, cnn_layer):
    return cnn_layer[2]*cnn_layer[4]*cnn_layer[5]*cnn_layer[6]

def infer_output_size(cnn_layer):
    W = cnn_layer[0]
    H = cnn_layer[1]
    Wstride = cnn_layer[9]
    Hstride = cnn_layer[10]
    R=cnn_layer[6]
    S=cnn_layer[5]
    P = (W-S+2*cnn_layer[7])/Wstride+1
    Q = (H-R+2*cnn_layer[8])/Hstride+1
    return (P, Q)

def infer_input_size(P_t, Q_t, cnn_layer):
    Wstride = cnn_layer[9]
    Hstride = cnn_layer[10]
    R=cnn_layer[6]
    S=cnn_layer[5]
    W_t = (P_t-1)*Wstride+R
    H_t = (Q_t-1)*Hstride+S
    return (W_t, H_t)

def fuse_layer(config, cnn_layers):
    initial_tile_P = 1
    initial_tile_Q = 1
    continue_fuse = False
    # groups of fused layers
    #[[cnn_layer_0, cnn_layer_1],[cnn_layer_2],[cnn_layer_3]]
    fused_groups = []
    fused_layers = []
    fusing_params = []
    # IWBufferSize = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] * config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['width'] / 1024 / 8 #buffer size in KB
    IWBufferSize = 2* 1024 #2MB
    IWPrecision = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['word-bits']
    print("InputWeightBuffer size (KB): "+ str(IWBufferSize))
    print("InputWeight Precision: "+str(IWPrecision))
    IWBufferSize = IWBufferSize*1024*8/IWPrecision
    print("InputWeightBuffer Max Elements: "+str(IWBufferSize))
    Q_t = initial_tile_Q
    P_t = initial_tile_P
    IWBufferRemain = IWBufferSize
    featureStorage = 0
    P_last, Q_last = infer_output_size(cnn_layers[-1])
    tile_count = div_ceil(P_last, P_t)*div_ceil(Q_last, Q_t)
    i = len(cnn_layers)-1
    while i >= 0:
        weightSize = get_weight_size(IWPrecision, cnn_layers[i])
        print("Layer "+str(i)+" weight size: " +str(weightSize))
        outputSize = P_t*Q_t*cnn_layers[i][4]
        (W_t, H_t) = infer_input_size(P_t, Q_t, cnn_layers[i])
        P_t = W_t
        Q_t = H_t
        inputSize = W_t*H_t*cnn_layers[i][2]
        featureStorage = max(outputSize, inputSize, featureStorage)
        if((IWBufferRemain-featureStorage)>=weightSize):
            
                
            IWBufferRemain-=weightSize
            fused_layers.insert(0, list(cnn_layers[i]))
            fusing_params.insert(0,[W_t, H_t, tile_count])
            # fused_layers.insert(0, [W_p, H_p, cnn_layers[i][2],cnn_layers[i][3]*tile_count, cnn_layers[i][4], cnn_layers[i][5], cnn_layers[i][6], cnn_layers[i][7], cnn_layers[i][8], cnn_layers[i][9], cnn_layers[i][10]])
            # print("fused tile input_size:" + " "+str(W_p) +" "+ str(H_p))
        else:
            #stop fuse
            #set initial values to P_t, Q_t, tile_count, IWBufferRemain, featureStorage
            #start another fused group
            P_t=initial_tile_P
            Q_t=initial_tile_Q
            W_t,H_t = infer_input_size(P_t, Q_t, cnn_layers[i])
            P_t = W_t
            Q_t = H_t
            P, Q = infer_output_size(cnn_layers[i])
            tile_count = div_ceil(P, P_t)*div_ceil(Q,Q_t)
            IWBufferRemain = IWBufferSize
            featureStorage = 0

            # if more than 1 layer to be fused, rewrite fused layers
            # if only 1 fused layer, do not rewrite
            # add fused layers to fused_groups
            print(fused_layers)
            if(len(fused_layers)>1):
                for j in range(len(fused_layers)-1,-1,-1):
                    temp_W_t = fusing_params[j][0]
                    temp_H_t = fusing_params[j][1]
                    temp_tile_count = fusing_params[j][2]
                    fused_layers[j][0] = temp_W_t
                    fused_layers[j][1] = temp_H_t
                    fused_layers[j][3] = cnn_layers[i][3]*temp_tile_count
                    fused_layers[j][7] = 0 #set padding to 0
                    fused_layers[j][8] = 0 #set padding to 0
            if(len(fused_layers)>0):
                fused_groups.insert(0, fused_layers)
            fused_layers = []
            fusing_params = []
            
            #try to add current layer to the new fused group with emptied buffer
            IWBufferRemain-=weightSize
            fused_layers.insert(0, list(cnn_layers[i]))
            fusing_params.insert(0,[W_t, H_t, tile_count])

            # print("terminate fusion. new fused tile input size: " + " "+str(W_p) +" "+ str(H_p))
        i = i-1

    if(len(fused_layers)>1):
        for j in range(len(fused_layers)-1,-1,-1):
            temp_W_t = fusing_params[j][0]
            temp_H_t = fusing_params[j][1]
            temp_tile_count = fusing_params[j][2]
            fused_layers[j][0] = temp_W_t
            fused_layers[j][1] = temp_H_t
            fused_layers[j][3] = cnn_layers[i][3]*temp_tile_count
            fused_layers[j][7] = 0 #set padding to 0
            fused_layers[j][8] = 0 #set padding to 0
    if(len(fused_layers)>0):
        fused_groups.insert(0, fused_layers)
    fused_layers = []
    fusing_params = []

    print("Fused groups: ")
    for i in range(len(fused_groups)):
        print(fused_groups[i])

    return fused_groups

