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
    W_o = (W-S+2*cnn_layer[7])/Wstride+1
    H_o = (H-R+2*cnn_layer[8])/Hstride+1
    return (W_o, H_o)

def infer_input_size(W_p, H_p, cnn_layer):
    Wstride = cnn_layer[9]
    Hstride = cnn_layer[10]
    R=cnn_layer[6]
    S=cnn_layer[5]
    W_p_i = (W_p-1)*Wstride+R
    H_p_i = (H_p-1)*Hstride+S
    return (W_p_i, H_p_i)

def fuse_layer(config, cnn_layers):
    initial_tile_H = 1
    initial_tile_W = 1
    continue_fuse = False
    # groups of fused layers
    #[[cnn_layer_0, cnn_layer_1],[cnn_layer_2],[cnn_layer_3]]
    fused_groups = []
    fused_layers = []
    fusing_params = []
    IWBufferSize = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['memory_depth'] * config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['memory_width'] / 1024 / 8 #buffer size in KB
    IWPrecision = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['word-bits']
    print("InputWeightBuffer size (KB): "+ str(IWBufferSize))
    print("InputWeight Precision: "+str(IWPrecision))
    IWBufferSize = IWBufferSize*1024*8/IWPrecision
    print("InputWeightBuffer Max Elements: "+str(IWBufferSize))
    W_p = initial_tile_W
    H_p = initial_tile_H
    IWBufferRemain = IWBufferSize
    featureStorage = 0
    tile_count = div_ceil(cnn_layers[-1][0], W_p)*div_ceil(cnn_layers[-1][1],H_p)
    i = len(cnn_layers)-1
    while i >= 0:
        weightSize = get_weight_size(IWPrecision, cnn_layers[i])
        print("Layer "+str(i)+" weight size: " +str(weightSize))
        outputSize = W_p*H_p*cnn_layers[i][4]
        (W_p, H_p) = infer_input_size(W_p, H_p, cnn_layers[i])
        inputSize = W_p*H_p*cnn_layers[i][2]
        featureStorage = max(outputSize, inputSize, featureStorage)
        if((IWBufferRemain-featureStorage)>=weightSize):
            
                
            IWBufferRemain-=weightSize
            fused_layers.insert(0, list(cnn_layers[i]))
            fusing_params.insert(0,[W_p, H_p, tile_count])
            # fused_layers.insert(0, [W_p, H_p, cnn_layers[i][2],cnn_layers[i][3]*tile_count, cnn_layers[i][4], cnn_layers[i][5], cnn_layers[i][6], cnn_layers[i][7], cnn_layers[i][8], cnn_layers[i][9], cnn_layers[i][10]])
            # print("fused tile input_size:" + " "+str(W_p) +" "+ str(H_p))
        else:
            #stop fuse
            #set initial values to W_p, H_p, tile_count, IWBufferRemain, featureStorage
            W_p=initial_tile_W
            H_p=initial_tile_H
            (W_o,H_o) = infer_output_size(cnn_layers[i])
            tile_count = div_ceil(W_o, W_p)*div_ceil(H_o,H_p)
            IWBufferRemain = IWBufferSize
            featureStorage = 0

            # if more than 1 layer to be fused, rewrite fused layers
            # if only 1 fused layer, do not rewrite
            # add fused layers to fused_groups
            print(fused_layers)
            if(len(fused_layers)>1):
                for j in range(len(fused_layers)-1,-1,-1):
                    temp_W_p = fusing_params[j][0]
                    temp_H_p = fusing_params[j][1]
                    temp_tile_count = fusing_params[j][2]
                    fused_layers[j][0] = temp_W_p
                    fused_layers[j][1] = temp_H_p
                    fused_layers[j][3] = cnn_layers[i][3]*temp_tile_count
                    fused_layers[j][7] = 0 #set padding to 0
                    fused_layers[j][8] = 0 #set padding to 0
            if(len(fused_layers)>0):
                fused_groups.insert(0, fused_layers)
            fused_layers = []
            fusing_params = []
            
            #try to fuse current layer with emptied buffer
            IWBufferRemain-=weightSize
            fused_layers.insert(0, list(cnn_layers[i]))
            fusing_params.insert(0,[W_p, H_p, tile_count])

            # print("terminate fusion. new fused tile input size: " + " "+str(W_p) +" "+ str(H_p))
        i = i-1

    if(len(fused_layers)>1):
        for j in range(len(fused_layers)-1,-1,-1):
            temp_W_p = fusing_params[j][0]
            temp_H_p = fusing_params[j][1]
            temp_tile_count = fusing_params[j][2]
            fused_layers[j][0] = temp_W_p
            fused_layers[j][1] = temp_H_p
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

