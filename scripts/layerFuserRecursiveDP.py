import layerFuserHelper as helper
def should_tile_last_layer_and_fuse(config, cnn_layers, buffer_size, initial_tile_P, initial_tile_Q):
    IWPrecision = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['word-bits']
    IWBufferSize = buffer_size* 1024 #buffer_size MB
    IWBufferSize = IWBufferSize*1024*8/IWPrecision
    featureStorage = 0
    i = len(cnn_layers)-1
    
    weightSize = helper.get_weight_size(IWPrecision, cnn_layers[i])
    if(IWBufferSize<weightSize):
        return False
    if(i==0):
        return False
    Q_t = initial_tile_Q
    P_t = initial_tile_P
    outputSize = P_t*Q_t*cnn_layers[i][4]
    (W_t, H_t) = helper.infer_input_size(P_t, Q_t, cnn_layers[i])
    inputSize = W_t*H_t*cnn_layers[i][2]
    featureStorage = max(outputSize, inputSize, featureStorage)
    upper_weight_size = helper.get_weight_size(IWPrecision, cnn_layers[i-1])
    if((IWBufferSize-featureStorage)<(upper_weight_size+weightSize)):
        return False
    return True

    

def fuse_layer_recursive(config, cnn_layers, buffer_size, initial_tile_P, initial_tile_Q, optimal_strategies_to_layer):
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
    weightSize = helper.get_weight_size(IWPrecision, cnn_layers[i])
    outputSize = P_t*Q_t*cnn_layers[i][4]
    optimal_strategies = optimal_strategies_to_layer[i]
    if((IWBufferRemain-featureStorage)<weightSize):
        # print("layer "+str(i)+" weight size exceeds buffer size, thus cannot be fused")   
        fused_groups.append([list(cnn_layers[i])])
        i = i-1
        # stop fuse
        if i>=0:
            upper_optimal_strategies = []
            output_P, output_Q = helper.infer_output_size(cnn_layers[i])
            for p in range(1, int(output_P)+1):
                    # print("p: " + str(p))
                # for q in range(2, 3):
                    print("should tile last layer and fuse: " + str(should_tile_last_layer_and_fuse(config, cnn_layers[0:i+1], buffer_size, p, p)))
                    if(should_tile_last_layer_and_fuse(config, cnn_layers[0:i+1], buffer_size, p, p) or (p==int(output_P))):
                        upper_optimal_strategies_pq = fuse_layer_recursive(config, cnn_layers[0:i], buffer_size, p, p, optimal_strategies_to_layer)
                        # print("upper_optimal_strategies: " + str(len(upper_optimal_strategies)))
                        for upper_strategy in upper_optimal_strategies_pq:
                            # print("update pareto front")
                            helper.updateParetoFront(upper_optimal_strategies, upper_strategy)
                            # print("upper_strategy: ")
                            # helper.printFusedGroup(upper_strategy)
                            # print("optimal_strategies: ")
                            # helper.printOptimalFusedGroups(optimal_strategies)
                    
            # print("optimal_strategies: " + str(len(optimal_strategies)))
            # helper.printOptimalFusedGroups(optimal_strategies)
            for strategy in upper_optimal_strategies:
                strategy.extend(fused_groups)
            for strategy in upper_optimal_strategies:
                helper.updateParetoFront(optimal_strategies, strategy)



    else:
        while (IWBufferRemain-featureStorage) >= helper.get_weight_size(IWPrecision, cnn_layers[i]) and i>=0:
            
            
            (W_t, H_t) = helper.infer_input_size(P_t, Q_t, cnn_layers[i])
            P_t = W_t
            Q_t = H_t
            inputSize = W_t*H_t*cnn_layers[i][2]
            featureStorage = max(outputSize, inputSize, featureStorage)
            
                
                    
            IWBufferRemain-=helper.get_weight_size(IWPrecision, cnn_layers[i])
            fused_layers.insert(0, list(cnn_layers[i]))
            fusing_params.insert(0,[W_t, H_t, tile_count])
            # fused_layers.insert(0, [W_p, H_p, cnn_layers[i][2],cnn_layers[i][3]*tile_count, cnn_layers[i][4], cnn_layers[i][5], cnn_layers[i][6], cnn_layers[i][7], cnn_layers[i][8], cnn_layers[i][9], cnn_layers[i][10]])
            # print("fused tile input_size:" + " "+str(W_p) +" "+ str(H_p))
            i = i-1

            # option 1: stop fuse

    
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
                    fused_layers[j][3] = fused_layers[j][3]*temp_tile_count
                    fused_layers[j][7] = 0 #set padding to 0
                    fused_layers[j][8] = 0 #set padding to 0
            if(len(fused_layers)>0):
                fused_groups.insert(0, fused_layers)
            # helper.printFusedGroup(fused_groups)
    
            if i>=0:
                upper_optimal_strategies = []
                output_P, output_Q = helper.infer_output_size(cnn_layers[i])
                for p in range(1, int(output_P)+1):
                        # print("p: " + str(p))
                    # for q in range(2, 3):
                        print("should tile last layer and fuse: " + str(should_tile_last_layer_and_fuse(config, cnn_layers[0:i+1], buffer_size, p, p)))
                        if(should_tile_last_layer_and_fuse(config, cnn_layers[0:i+1], buffer_size, p, p) or (p==int(output_P))):
                            upper_optimal_strategies_pq = fuse_layer_recursive(config, cnn_layers[0:i+1], buffer_size, p, p, optimal_strategies_to_layer)
                            # print("upper_optimal_strategies: " + str(len(upper_optimal_strategies)))
                            for upper_strategy in upper_optimal_strategies_pq:
                                # print("update pareto front")
                                helper.updateParetoFront(upper_optimal_strategies, upper_strategy)
                                # print("upper_strategy: ")
                                # helper.printFusedGroup(upper_strategy)
                                # print("optimal_strategies: ")
                                # helper.printOptimalFusedGroups(optimal_strategies)
                        
                # print("optimal_strategies: " + str(len(optimal_strategies)))
                # helper.printOptimalFusedGroups(optimal_strategies)
                for strategy in upper_optimal_strategies:
                    strategy.extend(fused_groups)
                for strategy in upper_optimal_strategies:
                    helper.updateParetoFront(optimal_strategies, strategy)

            else:
                helper.updateParetoFront(optimal_strategies, fused_groups)

    return optimal_strategies
   

def fuse_layer_recursive_start(config, cnn_layers, buffer_size):
    optimal_strategies_to_layer = []
    for i in range(len(cnn_layers)):
        optimal_strategies_to_layer.append([])
        
    optimal_strategies = []
    i = len(cnn_layers)
    output_P, output_Q = helper.infer_output_size(cnn_layers[i-1])
    for p in range(1, int(output_P)+1):
            # print("p: " + str(p))
        # for q in range(2, 3):
            print("should tile last layer and fuse: " + str(should_tile_last_layer_and_fuse(config, cnn_layers[0:i], buffer_size, p, p)))
            if(should_tile_last_layer_and_fuse(config, cnn_layers[0:i], buffer_size, p, p) or (p==int(output_P))):
                upper_optimal_strategies = fuse_layer_recursive(config, cnn_layers[0:i], buffer_size, p, p, optimal_strategies_to_layer)
                # print("upper_optimal_strategies: " + str(len(upper_optimal_strategies)))
                for upper_strategy in upper_optimal_strategies:
                    # print("update pareto front")
                    helper.updateParetoFront(optimal_strategies, upper_strategy)
                    # print("upper_strategy: ")
                    # helper.printFusedGroup(upper_strategy)
                    # print("optimal_strategies: ")
                    # helper.printOptimalFusedGroups(optimal_strategies)
            
    # print("optimal_strategies: " + str(len(optimal_strategies)))
    # helper.printOptimalFusedGroups(optimal_strategies)
    optimal_strategies_to_layer[i-1] = optimal_strategies
    return optimal_strategies