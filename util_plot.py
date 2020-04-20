import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#--------- figures: plotting -----------------------
def reverse_colormap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

#-------------------------------------------------------------------
#          Project 2: Transition matrix + Steady State plots
#-------------------------------------------------------------------

def save_norm_frequency(path_name,f):
    my_cmap = mpl.cm.pink
    #figsz_ = (10,4)
    x = [i for i in range(1,np.shape(f)[0]+1)]

    fig = plt.plot()
    
    plt.bar(x,f)
    #ax2.legend(['hidden state','buffer calculation'])
    #plt.set_title('normalized frequency')
    plt.xlabel('logkey')
    plt.ylabel('normalized frequency')
    plt.savefig(str(path_name)+'.eps',bbox_inches='tight')
    plt.close()
    return

def save_markov_chain(path_name,Phi,steady_state):

    my_cmap = mpl.cm.pink
    figsz_ = (10,4)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=figsz_)

    x = [i for i in range(1,np.shape(Phi)[0]+1)]

    img = ax1.imshow(Phi,cmap=my_cmap)
    #fig.colorbar(img,orientation='vertical')
    ax1.set_title('Markov Chain')
    ax1.set_ylabel('from-logkey')
    ax1.set_xlabel('to-logkey')
    
    ax2.bar(x,steady_state)
    #ax2.legend(['hidden state','buffer calculation'])
    ax2.set_title('steady states')
    ax2.set_xlabel('logkey')
    ax2.set_ylabel('steady state Prob.')
    fig.savefig(str(path_name)+'.eps',bbox_inches='tight')
    plt.close()
    return
    
    


"""
def save_fig_steady_states(path_name,P_ht_1,P_bf_1,phi_ht,phi_bf):
    #my_cmap = reverse_colormap(mpl.cm.bone) # can change the colormap type from here
    my_cmap = mpl.cm.pink
    figsz_ = (20,6)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=figsz_)

    x = [i for i in range(1,np.shape(P_ht_1)[0]+1)]

    img = ax1.imshow(P_ht_1,cmap=my_cmap)
    #fig.colorbar(img,orientation='vertical')
    ax1.set_title('P: hidden states')
    ax1.set_ylabel('from-logkey')
    ax1.set_xlabel('to-logkey')
    
    img = ax2.imshow(P_bf_1,cmap=my_cmap)
    #fig.colorbar(img,orientation='vertical')
    ax2.set_title('P: buffer calculation')
    ax2.set_ylabel('from-logkey')
    ax2.set_xlabel('to-logkey')
    
    ax3.plot(x,phi_ht,x,phi_bf)
    ax3.legend(['hidden state','buffer calculation'])
    ax3.set_title('hidden states')
    ax3.set_xlabel('logkey')
    ax3.set_ylabel('steady state P')
    fig.savefig(str(path_name)+'.eps',bbox_inches='tight')
    plt.close()
    return

def save_fig_markov_chain(path_name,P_ht,P_bf,logkey_index):    
    figsz_ = (16,7)
    fig, ax  = plt.subplots(2,3,figsize=figsz_)
    #my_cmap = reverse_colormap(mpl.cm.bone) # can change the colormap type from here
    my_cmap = mpl.cm.pink
    # Hidden state Markov chains
    P = P_ht[0][logkey_index,:,:]
    img = ax[0,0].imshow(P,cmap=my_cmap)
    #fig.colorbar(img)
    ax[0,0].set_title('Hidden state - step 1')

    P = P_ht[1][logkey_index,:,:]
    img = ax[0,1].imshow(P,cmap=my_cmap)
    #fig.colorbar(img)
    ax[0,1].set_title('Hidden state - step 2')

    P = P_ht[2][logkey_index,:,:]
    img = ax[0,2].imshow(P,cmap=my_cmap)
    #fig.colorbar(img)
    ax[0,2].set_title('Hidden state - step 3')

    # Buffer calculated  Markov chains
    P = P_bf[0][logkey_index,:,:]
    img = ax[1,0].imshow(P,cmap=my_cmap)
    #fig.colorbar(img)
    ax[1,0].set_title('Buffers - step 1')

    P = P_bf[1][logkey_index,:,:]
    img = ax[1,1].imshow(P,cmap=my_cmap)
    #fig.colorbar(img)
    ax[1,1].set_title('Buffers - step 2')

    P = P_bf[2][logkey_index,:,:]
    img = ax[1,2].imshow(P,cmap=my_cmap)
    #fig.colorbar(img)
    ax[1,2].set_title('Buffers - step 3')
    fig.savefig(str(path_name)+'.eps',bbox_inches='tight')
    plt.close()
    return


def saveTimeseriesFigure(path,name,data):
    figsz_ = (20,6)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=figsz_)

    tt = np.array(data[0])
    ax1.plot(np.arange(len(tt)),tt)
    ax1.set_ylabel('time-difference')
    ax1.set_title('TIME{}')
    
    tx = np.array(data[1])
    ax2.plot(np.arange(len(tx)),tx)
    ax2.set_ylabel('event-differnce')
    ax2.set_title('EVENTS{}')

    ty = np.array(data[2])
    ax3.plot(np.arange(len(ty)),ty)
    ax3.set_ylabel('IP-difference')
    ax3.set_title('IPs{}')
    fig.savefig(str(path)+str(name)+'.eps',bbox_inches='tight')
    return


def saveMatrixProfileFigure(path,name,data_org,data_test,matrix_profile,ylimit):
    figsz_ = (20,16)
    fig, ax = plt.subplots(3,3,sharex=True,figsize=figsz_)
    
    #--------- TIME ----------------
    time_org = data_org[0]
    time_test = data_test[0]
    mp_time = matrix_profile[0]
    
    ax[0,0].plot(np.arange(len(time_test)),time_org[0:len(time_test)])
    ax[0,0].set_ylabel('time-difference ( ms)')
    #ax[0,0].set_title('TIME')
    ax[0,0].set_title('CORRECT-TIME')
    
    ax[1,0].plot(np.arange(len(time_test)),time_test)
    ax[1,0].set_ylabel('time-difference ( ms)')
    #ax[0,0].set_title('TIME')
    ax[1,0].set_title('ANOMALY-TIME')
    
    ax[2,0].plot(np.arange(len(mp_time)),mp_time,color = 'red')
    ax[2,0].set_ylabel('MP-time')
    ax[2,0].set_title('MP-TIME')
    ax[2,0].set_xlabel('time-units')
    ax[2,0].set_ylim(ylimit[0],ylimit[1])
    
    #----------- EVENT --------------
    event_org = data_org[1]
    event_test = data_test[1]
    mp_event = matrix_profile[1]
    
    ax[0,1].plot(np.arange(len(event_test)),event_org[0:len(event_test)])
    ax[0,1].set_ylabel('event-difference')
    ax[0,1].set_title('CORRECT-EVENT')
    
    ax[1,1].plot(np.arange(len(event_test)),event_test)
    ax[1,1].set_ylabel('event-difference')
    ax[1,1].set_title('ANOMALY-EVENT')
    
    ax[2,1].plot(np.arange(len(mp_event)),mp_event,color = 'red')
    ax[2,1].set_ylabel('MP-event')
    ax[2,1].set_title("MP-EVENT")
    ax[2,1].set_xlabel('time-units')
    ax[2,1].set_ylim(ylimit[0],ylimit[1])
    
    #------------ IP --------------------
    ip_org = data_org[2]
    ip_test = data_test[2]
    mp_ip = matrix_profile[2]
    
    ax[0,2].plot(np.arange(len(ip_test)),ip_org[0:len(ip_test)])
    ax[0,2].set_ylabel('IP-difference')
    ax[0,2].set_title('CORRECT-IP')
    
    ax[1,2].plot(np.arange(len(ip_test)),ip_test)
    ax[1,2].set_ylabel('IP-difference')
    ax[1,2].set_title('ANOMALY-IP')
    
    ax[2,2].plot(np.arange(len(mp_ip)),mp_ip,color = 'red')
    ax[2,2].set_ylabel('MP-ip')
    ax[2,2].set_title("MP-IP")
    ax[2,2].set_xlabel('time-units')
    ax[2,2].set_ylim(ylimit[0],ylimit[1])
    #------------------------------------
    fig.savefig(str(path)+str(name),bbox_inches='tight')
    return

def savefftFigure(path,name,data_org,data_test,fft_data):
    figsz_ = (20,16)
    fig, ax = plt.subplots(3,3,sharex=False,figsize=figsz_)
    
    #--------- TIME ----------------
    time_org = data_org[0]
    time_test = data_test[0]
    mp_time = fft_data[0]
    
    ax[0,0].plot(np.arange(len(time_test)),time_org[0:len(time_test)])
    ax[0,0].set_ylabel('time-difference ( ms)')
    #ax[0,0].set_title('TIME')
    ax[0,0].set_title('CORRECT-TIME')
    
    ax[1,0].plot(np.arange(len(time_test)),time_test)
    ax[1,0].set_ylabel('time-difference ( ms)')
    #ax[0,0].set_title('TIME')
    ax[1,0].set_title('ANOMALY-TIME')
    
    ax[2,0].plot(np.arange(len(mp_time)),mp_time,color = 'green')
    ax[2,0].set_ylabel('time-difference (ms)')
    ax[2,0].set_title('FFT-TIME')
    ax[2,0].set_xlabel('time-units')
    #ax[2,0].set_ylim(ylimit[0],ylimit[1])
    
    #----------- EVENT --------------
    event_org = data_org[1]
    event_test = data_test[1]
    mp_event = fft_data[1]
    
    ax[0,1].plot(np.arange(len(event_test)),event_org[0:len(event_test)])
    ax[0,1].set_ylabel('event-difference')
    ax[0,1].set_title('CORRECT-EVENT')
    
    ax[1,1].plot(np.arange(len(event_test)),event_test)
    ax[1,1].set_ylabel('event-difference')
    ax[1,1].set_title('ANOMALY-EVENT')
    
    ax[2,1].plot(np.arange(len(mp_event)),mp_event,color = 'green')
    ax[2,1].set_ylabel('event-difference')
    ax[2,1].set_title("FFT-EVENT")
    ax[2,1].set_xlabel('time-units')
    #ax[2,1].set_ylim(ylimit[0],ylimit[1])
    
    #------------ IP --------------------
    ip_org = data_org[2]
    ip_test = data_test[2]
    mp_ip = fft_data[2]
    
    ax[0,2].plot(np.arange(len(ip_test)),ip_org[0:len(ip_test)])
    ax[0,2].set_ylabel('IP-difference')
    ax[0,2].set_title('CORRECT-IP')
    
    ax[1,2].plot(np.arange(len(ip_test)),ip_test)
    ax[1,2].set_ylabel('IP-difference')
    ax[1,2].set_title('ANOMALY-IP')
    
    ax[2,2].plot(np.arange(len(mp_ip)),mp_ip,color = 'green')
    ax[2,2].set_ylabel('IP-difference')
    ax[2,2].set_title("FFT-IP")
    ax[2,2].set_xlabel('time-units')
    #ax[2,2].set_ylim(ylimit[0],ylimit[1])
    #------------------------------------
    fig.savefig(str(path)+str(name),bbox_inches='tight')
    return

"""
