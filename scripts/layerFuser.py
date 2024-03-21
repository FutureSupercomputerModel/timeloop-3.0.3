import yaml
import layerFuserHelper as helper


def fuse_layer_PQ(config, cnn_layers, buffer_size, initial_tile_P, initial_tile_Q):
    # initial_tile_P = 1
    # initial_tile_Q = 1
    continue_fuse = False
    # groups of fused layers
    #[[cnn_layer_0, cnn_layer_1],[cnn_layer_2],[cnn_layer_3]]
    fused_groups = []
    fused_layers = []
    fusing_params = []
    # IWBufferSize = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] * config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['width'] / 1024 / 8 #buffer size in KB
    IWBufferSize = buffer_size* 1024 #buffer_size MB
    IWPrecision = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['word-bits']
    # print("InputWeightBuffer size (KB): "+ str(IWBufferSize))
    # print("InputWeight Precision: "+str(IWPrecision))
    IWBufferSize = IWBufferSize*1024*8/IWPrecision
    # print("InputWeightBuffer Max Elements: "+str(IWBufferSize))
    Q_t = initial_tile_Q
    P_t = initial_tile_P
    IWBufferRemain = IWBufferSize
    featureStorage = 0
    P_last, Q_last = helper.infer_output_size(cnn_layers[-1])
    tile_count = helper.div_ceil(P_last, P_t)*helper.div_ceil(Q_last, Q_t)
    i = len(cnn_layers)-1
    while i >= 0:
        weightSize = helper.get_weight_size(IWPrecision, cnn_layers[i])
        # print("Layer "+str(i)+" weight size: " +str(weightSize))
        outputSize = P_t*Q_t*cnn_layers[i][4]
        (W_t, H_t) = helper.infer_input_size(P_t, Q_t, cnn_layers[i])
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
            W_t,H_t = helper.infer_input_size(P_t, Q_t, cnn_layers[i])
            P, Q = helper.infer_output_size(cnn_layers[i])
            tile_count = helper.div_ceil(P, P_t)*helper.div_ceil(Q,Q_t) 
            P_t = W_t
            Q_t = H_t
            
            IWBufferRemain = IWBufferSize
            featureStorage = 0

            # if more than 1 layer to be fused, rewrite fused layers
            # if only 1 fused layer, do not rewrite
            # add fused layers to fused_groups
            # print(fused_layers)
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



    return fused_groups








def fuse_layer(config, cnn_layers, buffer_size):
    i = len(cnn_layers)-1
    output_P, output_Q = helper.infer_output_size(cnn_layers[i])
    total_macs = 0
    total_offchip_access = 0
    all_fused_groups = []
    for p in range(1, int(output_P)+1):
        for q in range(1, int(output_Q)+1):
            # print("Fusing with tile size: "+str(p)+" "+str(q))
            fused_groups = fuse_layer_PQ(config, cnn_layers, buffer_size, p, q)
            print("fused group "+ str(p) +" "+ str(q) + ": total macs="+ str(helper.get_total_macs(fused_groups)) + " total offchip access="+ str(helper.get_total_offchip_access(fused_groups)))
            helper.printFusedGroup(fused_groups)

            helper.updateParetoFront(all_fused_groups, fused_groups)

    helper.printOptimalFusedGroups(all_fused_groups)
    return all_fused_groups
            
