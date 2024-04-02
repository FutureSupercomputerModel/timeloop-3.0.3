import copy
import matplotlib.pyplot as plt

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
    
def infer_prev_layer_output_size(W_t, H_t, prev_pooling_layer):
    P_t = W_t * prev_pooling_layer[0]
    Q_t = H_t * prev_pooling_layer[1]
    return (P_t, Q_t)

def get_total_macs(fused_groups):
    total_macs = 0
    for i in range(len(fused_groups)):
        for j in range(len(fused_groups[i])):
            P,Q = infer_output_size(fused_groups[i][j])
            # PQCNKSR
            total_macs += P*Q*fused_groups[i][j][2]*fused_groups[i][j][3]*fused_groups[i][j][4]*fused_groups[i][j][5]*fused_groups[i][j][6]
    return total_macs

def get_total_offchip_access(fused_groups):
    total_offchip_access = 0
    for i in range(len(fused_groups)):
        first_layer = fused_groups[i][0]
        last_layer = fused_groups[i][-1]
        total_offchip_access+=first_layer[0]*first_layer[1]*first_layer[2]*first_layer[3]
        last_layer_P, last_layer_Q = infer_output_size(last_layer)
        total_offchip_access+=last_layer_P*last_layer_Q*last_layer[3]*last_layer[4]
    return total_offchip_access

# is a_, b_ strictly worse than a, b?
def isStrictlyWorse(a,b,a_,b_):
    if a_>a and b_>b:
        return True
    return False

# is a_, b_ strictly better than a, b?
def isStrictlyBetter(a,b,a_,b_):
    if a_<a and b_<b:
        return True
    return False

def printStrategy(strategy):
    for i in range(len(strategy)):
        print("\t\t"+str(strategy[i]))

def printStrategies(strategies):
    print("\tPareto Optimal Strategies: ")
    for i in range(len(strategies)):
        print("\tstrategy "+ str(i) + ": total macs="+ str(get_total_macs(strategies[i])) + " total offchip access="+ str(get_total_offchip_access(strategies[i])))
        printStrategy(strategies[i])

def summarizeStrategies(strategies, sram_size):
    macs = []
    offchip_access = []
    for i in range(len(strategies)):
        total_macs = get_total_macs(strategies[i])
        total_offchip_access = get_total_offchip_access(strategies[i])
        macs.append(total_macs)
        offchip_access.append(total_offchip_access)
    
    plt.scatter(macs, offchip_access)
    plt.xlabel("Total MACs")
    plt.ylabel("Total Off-chip Access")
    plt.show()
    plt.savefig(f"macs_vs_offchip_access_{sram_size}.png")

def updateParetoFront(all_fused_groups, this_group):
    if all_fused_groups == []:
        all_fused_groups.append(copy.deepcopy(this_group))
        return
    # print("all_fused_groups: "+str(len(all_fused_groups)))
    # print("all strategies: ")
    # for i in range(len(all_fused_groups)):
    #     print(str(get_total_macs(all_fused_groups[i])) + " " + str(get_total_offchip_access(all_fused_groups[i])))
    this_total_macs = get_total_macs(this_group)
    this_total_offchip_access = get_total_offchip_access(this_group)
    # print("this strategy: "+str(this_total_macs)+" "+str(this_total_offchip_access))
    new_all_fused_groups = []
    for i in range(len(all_fused_groups)):
        # Pareto optimal if total_macs is less than the total_macs of any other group, 
        # or total_offchip_access is less than the total_offchip_access of any other group, 
        # or both are equal to any other group
        total_macs_to_be_compared = get_total_macs(all_fused_groups[i])
        total_offchip_access_to_be_compared =get_total_offchip_access(all_fused_groups[i])
        # add to new_all_fused_groups if all_fused_groups[i] is not strictly worse than this_group
        if not isStrictlyWorse(this_total_macs, this_total_offchip_access, total_macs_to_be_compared, total_offchip_access_to_be_compared):
            new_all_fused_groups.append(copy.deepcopy(all_fused_groups[i]))
    
    append = True
    # add this_group if it is not strictly worse than any group in all_fused_groups
    for i in range(len(all_fused_groups)):
        total_macs_to_be_compared = get_total_macs(all_fused_groups[i])
        total_offchip_access_to_be_compared = get_total_offchip_access(all_fused_groups[i])
        if isStrictlyWorse(total_macs_to_be_compared, total_offchip_access_to_be_compared, this_total_macs, this_total_offchip_access):
            append = False
            break

    if append:
        new_all_fused_groups.append(copy.deepcopy(this_group))
    all_fused_groups.clear()
    all_fused_groups.extend(new_all_fused_groups)
    # print(len(all_fused_groups))