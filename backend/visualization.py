# visualization.py -> visualize a few samples from your dataset (all three axes) and from both classes (‘walking’ and ‘jumping’)

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns


with h5py.File('datafile.hdf5', 'r') as hdfr:
    walk_s = np.array(hdfr.get('Salvina/data\w1_salvina.csv'))
    jump_s = (np.array(hdfr.get('Salvina/data\j1_salvina.csv')))
    walk_j = (np.array(hdfr.get('Jonah/data\w1_jonah.csv')))
    jump_j = (np.array(hdfr.get('Jonah/data\j1_jonah.csv')))
    walk_l = (np.array(hdfr.get('Liam/data\w1_liam.csv')))
    jump_l = (np.array(hdfr.get('Liam/data\j1_liam.csv')))
    data = np.array(hdfr.get('dataset/Test/x_test'))
    dflabel = np.array(hdfr.get('dataset/Test/y_test'))

data = pd.DataFrame(data, columns=['x_max', 'y_max', 'z_max','x_min', 'y_min', 'z_min','x_range', 'y_range', 'z_range','x_mean', 'y_mean', 'z_mean',
                                                    'x_median', 'y_median', 'z_median','x_std', 'y_std', 'z_std', 'x_skew', 'y_skew', 'z_skew','x_kurt', 'y_kurt', 'z_kurt',
                                                    'x_variance', 'y_variance', 'z_variance', 'x_rms','y_rms', 'z_rms', 'coeffxy', 'coeffyz', 'coeffxz'])
data['label'] = dflabel

walk_arrays, jump_arrays = [],[]
walk_arrays.append(walk_s)
walk_arrays.append(walk_j)
walk_arrays.append(walk_l)
jump_arrays.append(jump_s)
jump_arrays.append(jump_j)
jump_arrays.append(jump_l)
names = ['Salvina', 'Jonah', 'Liam']
walk_length = len(walk_s) + len(walk_j) + len(walk_l)
jump_length = len(jump_s) + len(jump_j) + len(jump_l)

# acceleration graphs for all members 
fig, ax = plt.subplots(3, 2, figsize=(8, 6))
row = 0
for member_array, name in zip(walk_arrays, names):
    col = 0
    if row < 3: 
        time = member_array[:1000, 0]
        ax[row][col].plot(time, member_array[:1000, 1], label='x')
        ax[row][col].plot(time, member_array[:1000, 2], label='y')
        ax[row][col].plot(time,member_array[:1000, 3], label='z')
        ax[row][col].set_title('Walking Data '+ '(' + name+')')
        ax[row][col].set_ylabel('Acceleration (m/s^2)')
        ax[row][col].set_xlabel('Time (s)')
        ax[row][col].legend()
        row = row + 1
    
row = 0
for member_array, name in zip(jump_arrays, names):
    col = 1
    if row < 3: 
        time = member_array[:1000, 0]
        ax[row][col].plot(time, member_array[:1000, 1], label='x')
        ax[row][col].plot(time, member_array[:1000, 2], label='y')
        ax[row][col].plot(time,member_array[:1000, 3], label='z')
        ax[row][col].set_title('Jumping Data '+ '(' + name+')')
        ax[row][col].set_ylabel('Acceleration (m/s^2)')
        ax[row][col].set_xlabel('Time (s)')
        ax[row][col].legend()
        row = row + 1

fig.subplots_adjust(hspace=0.7)
plt.show()    

#Create bar chart of sample counts
walking_count = walk_length
jumping_count = jump_length
fig, ax = plt.subplots()
bargraph = ax.bar(['Walking', 'Jumping'], [walking_count, jumping_count])
ax.set_ylabel('Number of Samples')
for bars in bargraph:
    height = bars.get_height()
    ax.text(x=(bars.get_x() + bars.get_width() / 2) , y =height+1 , s=f"{height}" )
plt.show()


#boxplot
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
row = 0
for member_array, name in zip(walk_arrays, names):
    col = 0
    if row < 3: 
        ax[row][col].boxplot([member_array[:1000, 1],  member_array[:1000, 2],  member_array[:1000, 3]], labels=['x', 'y', 'z'])
        ax[row][col].set_title('Walking Data Boxplot'+ '(' + name+')')
        ax[row][col].set_ylabel('Acceleration (m/s^2)')
        ax[row][col].set_xlabel('Axis')
        row = row + 1

row = 0
for member_array, name in zip(jump_arrays, names):
    col = 1
    if row < 3: 
        ax[row][col].boxplot([member_array[:1000, 1],  member_array[:1000, 2],  member_array[:1000, 3]], labels=['x', 'y', 'z'])
        ax[row][col].set_title('Jumping Data Boxplot'+ '(' + name+')')
        ax[row][col].set_ylabel('Acceleration (m/s^2)')
        ax[row][col].set_xlabel('Axis')
        row = row + 1

fig.subplots_adjust(hspace=0.5)
plt.show()


# # Histograms
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
row = 0
for member_array, name in zip(walk_arrays, names):
    if row < 3: 
        # Histogram for x-coordinate
        ax[row][0].hist(member_array[:, 1], bins=50)
        ax[row][0].set_title('Walking Histogram' + '('+ name+')')
        ax[row][0].set_xlabel('Acceleration (m/s^2)')
        ax[row][0].set_ylabel('Count for x-coordinate')

        # Histogram for y-coordinate
        ax[row][1].hist(member_array[:, 2], bins=50)
        ax[row][1].set_title('Walking Histogram' + '('+ name+')')
        ax[row][1].set_xlabel('Acceleration (m/s^2)')
        ax[row][1].set_ylabel('Count for y-coordinate')

        # Histogram for z-coordinate
        ax[row][2].hist(member_array[:, 3], bins=50)
        ax[row][2].set_title('Walking Histogram' + '('+ name+')')
        ax[row][2].set_xlabel('Acceleration (m/s^2)')
        ax[row][2].set_ylabel('Count for z coordinate')
        row = row + 1

fig.subplots_adjust(hspace=0.5)
plt.show()

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
row = 0
for member_array, name in zip(jump_arrays, names):
    if row < 3: 
        # Histogram for x-coordinate
        ax[row][0].hist(member_array[:, 1], bins=50)
        ax[row][0].set_title('Jumping Histogram' + '('+ name+')')
        ax[row][0].set_xlabel('Acceleration (m/s^2)')
        ax[row][0].set_ylabel('Count for x-coordinate')

        # Histogram for y-coordinate
        ax[row][1].hist(member_array[:, 2], bins=50)
        ax[row][1].set_title('Jumping Histogram' + '('+ name+')')
        ax[row][1].set_xlabel('Acceleration (m/s^2)')
        ax[row][1].set_ylabel('Count for y-coordinate')

        # Histogram for z-coordinate
        ax[row][2].hist(member_array[:, 3], bins=50)
        ax[row][2].set_title('Jumping Histogram' + '('+ name+')')
        ax[row][2].set_xlabel('Acceleration (m/s^2)')
        ax[row][2].set_ylabel('Count for z coordinate')
        row = row + 1


fig.subplots_adjust(hspace=0.5)
plt.show()

# differences between jumpin and walking, look at distribution of each feature
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
num_cols = data.columns
print(num_cols)

for col in num_cols:
    column_name = col.title().replace('_', ' ')
    title = 'Distribution of ' + column_name
    sns.boxplot(x=data[col],y=data['label'],data=data,orient='h',showmeans=True)
    plt.xlabel(column_name, fontsize = 12)
    plt.title(title, fontsize = 14, pad = 10)
    plt.show()
