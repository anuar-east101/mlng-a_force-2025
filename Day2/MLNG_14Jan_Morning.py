
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

X = np.linspace(0,10,11)
Y = 10*X + np.random.rand(len(X))*30

fig, ax1 = plt.subplots(1,1)
ax1.plot(X, Y, 'ob')

# m = 10

# Y_pred = m*X

# rmse = np.sqrt( sum((Y-Y_pred)**2) / len(Y) )

ms = np.linspace(0,20,21)
store_rmse = []

for m in ms:
    Y_pred = m*X

    rmse = np.sqrt( sum((Y-Y_pred)**2) / len(Y) )
    store_rmse.append(rmse)
    ax1.plot(X,Y_pred, 'grey', alpha=0.5)

min_rmse_loc = np.argmin(np.array(store_rmse))
ax1.plot(X,X*ms[min_rmse_loc], 'g')
    
fig, ax2 = plt.subplots(1,1)
ax2.plot(ms, store_rmse, 'ob')

#%%

grad_desc = pd.DataFrame(columns=['Iteration', 'm_old', 'm_now', 'rmse'])

lr = 0.5
m_init = 19
iter_no = 30

m_now = m_init
Y_pred = X*m_now     
rmse = np.sqrt(sum((Y_pred-Y)**2)/len(Y))
grad_desc.loc[0] = [0, 'NA', m_now, rmse]

m_now = 2
Y_pred = X*m_now     
rmse = np.sqrt(sum((Y_pred-Y)**2)/len(Y))
grad_desc.loc[1] = [1, 'NA', m_now, rmse]

for ii in range(2,iter_no):
    grad = (grad_desc.loc[ii-1]['rmse']-grad_desc.loc[ii-2]['rmse'])  \
           /(grad_desc.loc[ii-1]['m_now']-grad_desc.loc[ii-2]['m_now'])
    m_old = grad_desc.loc[ii-1]['m_now']
    m_now = m_old - lr*grad 
    Y_pred = X*m_now  
    rmse = np.sqrt(sum((Y_pred-Y)**2)/len(Y))
    grad_desc.loc[len(grad_desc)] = [ii, m_old, m_now, rmse]

#%%
import imageio

images = []
fig, ax3 = plt.subplots(1,3, figsize=(12,4))
ax3[0].plot(ms, store_rmse, '-b', alpha=0.5)
ax3[0].set_xlabel('M')
ax3[0].set_ylabel('RMSE')

ax3[1].set_title('RMSE')
ax3[1].set_xlim(0,iter_no)
ax3[1].set_ylim(0,100)
ax3[1].set_xlabel('Iteration')
ax3[1].set_ylabel('RMSE')

ax3[2].plot(X, Y, 'ob')

for ii in range(iter_no):
    fig, ax3 = plt.subplots(1,3, figsize=(12,4))
    ax3[0].plot(ms, store_rmse, '-b', alpha=0.5)
    ax3[0].set_xlabel('M')
    ax3[0].set_ylabel('RMSE')
    
    ax3[1].set_title('RMSE')
    ax3[1].set_xlim(0,iter_no)
    ax3[1].set_ylim(0,100)
    ax3[1].set_xlabel('Iteration')
    ax3[1].set_ylabel('RMSE')
    
    ax3[2].plot(X, Y, 'ob')     
    ax3[0].plot(grad_desc['m_now'].loc[:ii], grad_desc['rmse'].loc[:ii],'--or', alpha=0.8)
    ax3[0].set_title('Iteration no:' + str(ii))
    
    ax3[1].plot(grad_desc['Iteration'].loc[:ii], grad_desc['rmse'].loc[:ii],'--or', alpha=0.8)
    
    ax3[2].plot(X, X*grad_desc['m_now'][ii], 'g')
    
    file_name = 'grad_desc_'+str(ii)+'.png'
    plt.savefig(file_name)
    
    images.append(imageio.imread(file_name))

# save the images as a gif       
imageio.mimsave('grad_desc.gif', images)
    