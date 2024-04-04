import layerFuserHelper as helper
import copy
def should_tile_last_layer_and_fuse(config, cnn_layers, buffer_size, initial_tile_P, initial_tile_Q):
    # IWPrecision = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['word-bits']
    IWPrecision = 8
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
    featureStorage = max(outputSize+inputSize, featureStorage)
    upper_weight_size = helper.get_weight_size(IWPrecision, cnn_layers[i-1])
    if((IWBufferSize-featureStorage)<(upper_weight_size+weightSize)):
        return False
    return True

    

def fuse_layer_recursive(config, cnn_layers, pooling_layers, buffer_size, optimal_strategies_to_layer):
    print("calling fuse_layer_recursive "+ str(len(cnn_layers)-1))  
    continue_fuse = False
    # groups of fused layers
    #[[cnn_layer_0, cnn_layer_1],[cnn_layer_2],[cnn_layer_3]]
    
    # IWBufferSize = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] * config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['width'] / 1024 / 8 #buffer size in KB
    IWBufferSize = buffer_size* 1024 #buffer_size MB
    # IWPrecision = config['arch']['subtree'][0]['subtree'][0]['local'][0]['attributes']['word-bits']
    IWPrecision = 8
    # print("InputWeightBuffer size (KB): "+ str(IWBufferSize))
    # print("InputWeight Precision: "+str(IWPrecision))
    IWBufferSize = IWBufferSize*1024*8/IWPrecision
    # print("InputWeightBuffer Max Elements: "+str(IWBufferSize))
    
    optimal_strategies = optimal_strategies_to_layer[len(cnn_layers)-1]
    # strategy to fuse last layer
    P_last, Q_last = helper.infer_output_size(cnn_layers[-1])
    for initial_P in range(1, int(P_last)+1):
            initial_Q = initial_P
            i = len(cnn_layers)-1
            if(should_tile_last_layer_and_fuse(config, cnn_layers[0:i+1], buffer_size, initial_P, initial_Q)):
                fused_groups = []
                fused_layers = []
                fusing_params = []
                Q_t = initial_P
                P_t = initial_Q
                IWBufferRemain = IWBufferSize
                featureStorage = 0
                
                tile_count = helper.div_ceil(P_last, P_t)*helper.div_ceil(Q_last, Q_t)
                outputSize = P_t*Q_t*cnn_layers[i][4]
                (W_t, H_t) = helper.infer_input_size(P_t, Q_t, cnn_layers[i])
                inputSize = W_t*H_t*cnn_layers[i][2]
                featureStorage = max(outputSize+inputSize, featureStorage)
                while (IWBufferRemain-featureStorage) >= helper.get_weight_size(IWPrecision, cnn_layers[i]) and i>=0:
                    
                    # print("start while loop with p, i: "+ str(initial_P)+" "+ str(i))
                    
                    
                        
                    IWBufferRemain-=helper.get_weight_size(IWPrecision, cnn_layers[i])
                   
                    fused_layers.insert(0, copy.deepcopy(list(cnn_layers[i])))
                    fusing_params.insert(0,copy.deepcopy([W_t, H_t, tile_count]))
                    i = i-1
                    if i>=0: 
                        P_t, Q_t = helper.infer_prev_layer_output_size(W_t, H_t, pooling_layers[i]) 
                        (W_t, H_t) = helper.infer_input_size(P_t, Q_t, cnn_layers[i])
                        outputSize = P_t*Q_t*cnn_layers[i][4]
                        inputSize = W_t*H_t*cnn_layers[i][2]
                        featureStorage = max(outputSize+inputSize, featureStorage)
                    

                    # rewrite fused layers
                    
                        
                    temp_W_t = fusing_params[0][0]
                    temp_H_t = fusing_params[0][1]
                    temp_tile_count = fusing_params[0][2]
                    fused_layers[0][0] = temp_W_t
                    fused_layers[0][1] = temp_H_t
                    fused_layers[0][3] = fused_layers[0][3]*temp_tile_count
                    fused_layers[0][7] = 0 #set padding to 0
                    fused_layers[0][8] = 0 #set padding to 0
                    
                    fused_groups=copy.deepcopy([fused_layers])

                    # option 1: stop fuse
                    # get upper optimal strategies and append current fused group
                    if len(fused_layers)<=1:
                        continue
                    if i>=0:
                        upper_optimal_strategies = []
                        if(optimal_strategies_to_layer[i] == []):   
                            upper_optimal_strategies = fuse_layer_recursive(config, cnn_layers[0:i+1], pooling_layers[0:i+1], buffer_size, optimal_strategies_to_layer)
                        else:
                            upper_optimal_strategies = optimal_strategies_to_layer[i]
                        # print("optimal_strategies: " + str(len(optimal_strategies)))
                        # helper.printOptimalFusedGroups(optimal_strategies)
                        for strategy in upper_optimal_strategies:
                            # print("combine results:")
                            # print(strategy)
                            # print(fused_groups)
                            combined_strategy = copy.deepcopy(strategy)
                            combined_strategy.extend(copy.deepcopy(fused_groups))
                            # print(strategy)
                        
                            helper.updateParetoFront(optimal_strategies, combined_strategy)
                        
                        # print("upper_optimal_strategies: " + str(len(upper_optimal_strategies)))
                        # helper.printStrategies(upper_optimal_strategies)
                        # print("optimal_strategies: " + str(len(optimal_strategies)))
                        # helper.printStrategies(optimal_strategies)

                    else:
                        # print("update optimal_strategies with i=-1:")
                        # print(optimal_strategies)
                        # print(fused_groups)
                        helper.updateParetoFront(optimal_strategies, fused_groups)
                        # print(optimal_strategies)
                    
                    
                # print("optimal strategy at end of this loop:")
                # print(optimal_strategies)
                    # option 2: continue fuse
    

    # strategy to not fuse last layer 
    fused_groups = []
    i = len(cnn_layers)-1
    fused_groups.append([list(cnn_layers[i])])
    i = i-1
    # stop fuse
    if i>=0:
        upper_optimal_strategies = []
        if(optimal_strategies_to_layer[i] == []):  
                 
            upper_optimal_strategies = fuse_layer_recursive(config, cnn_layers[0:i+1], pooling_layers[0:i+1], buffer_size, optimal_strategies_to_layer)
            # print("upper_optimal_strategies: " + str(len(upper_optimal_strategies)))
        else:
            upper_optimal_strategies = optimal_strategies_to_layer[i]      
        # print("optimal_strategies: " + str(len(optimal_strategies)))
        # helper.printOptimalFusedGroups(optimal_strategies)
        for strategy in upper_optimal_strategies:
            combined_strategy = copy.deepcopy(strategy)
            combined_strategy.extend(copy.deepcopy(fused_groups))
            helper.updateParetoFront(optimal_strategies, combined_strategy)
    else:
        helper.updateParetoFront(optimal_strategies, fused_groups)

        # print("upper_optimal_strategies: " + str(len(upper_optimal_strategies)))
        # helper.printStrategies(upper_optimal_strategies)
        # print("optimal_strategies: " + str(len(optimal_strategies)))
        # helper.printStrategies(optimal_strategies)
    # print(optimal_strategies)
    # print(optimal_strategies_to_layer)
    print("call return")
    return optimal_strategies
   

def fuse_layer_recursive_start(config, cnn_layers, pooling_layers, buffer_size):
    optimal_strategies_to_layer = []
    for i in range(len(cnn_layers)):
        optimal_strategies_to_layer.append([])
    optimal_strategies = fuse_layer_recursive(config, cnn_layers, pooling_layers, buffer_size, optimal_strategies_to_layer)
    for i in range(len(cnn_layers)):
        print("optimal_strategies to layer "+str(i)+":")
        helper.printStrategies(optimal_strategies_to_layer[i])
    return optimal_strategies