
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# Load threshold results
threshold_data = pd.read_pickle('/Users/asbjorn/Desktop/results_bachelor/thresholdingresults.pkl')
threshold_intervals_lower = []
threshold_intervals_higher = []
threshold_means = []

threshold_data = [data[0] for data in list(threshold_data.values())]

for i in range(0, len(threshold_data), 5):

  fold_acc = threshold_data[i:i+5]
  mean = sum(fold_acc) / len(fold_acc)
  sigma = np.std(fold_acc)
  std_err = sigma/np.sqrt(5)
  
  threshold_intervals_lower.append(mean-std_err)
  threshold_intervals_higher.append(mean+std_err)
  threshold_means.append(mean)

# Compute ranges for 3D-RPL

rpl_data = pd.read_csv('/Users/asbjorn/Desktop/results_bachelor/3drpl_freeze.csv')

names = np.unique(list(rpl_data.Name))
rpl_intervals_lower = []
rpl_intervals_higher = []
rpl_means = []

for name in names:
  result = rpl_data.loc[rpl_data['Name'] == name, 'test_dice_metric_value'].values
  mean = np.mean(result)
  sigma = np.std(result)
  std_err = sigma/np.sqrt(5)
  rpl_intervals_lower.append(mean-std_err)
  rpl_intervals_higher.append(mean+std_err)
  rpl_means.append(mean)


random_data = pd.read_csv('/Users/asbjorn/Desktop/results_bachelor/random.csv')

names = np.unique(list(random_data.Name))
random_intervals_lower = []
random_intervals_higher = []
random_means = []

for name in names:
  result = random_data.loc[random_data['Name'] == name, 'test_dice_metric_value'].values
  mean = np.mean(result)
  sigma = np.std(result)
  std_err = sigma/np.sqrt(5)
  random_intervals_lower.append(mean-std_err)
  random_intervals_higher.append(mean+std_err)
  random_means.append(mean)
  
transfer_data = pd.read_csv('/Users/asbjorn/Desktop/results_bachelor/transfer_freeze.csv')


names = np.unique(list(transfer_data.Name))
transfer_intervals_lower = []
transfer_intervals_higher = []
transfer_means = []

for name in names:
  result = transfer_data.loc[transfer_data['Name'] == name, 'test_dice_metric_value'].values
  mean = np.mean(result)
  sigma = np.std(result)
  std_err = sigma/np.sqrt(5)
  transfer_intervals_lower.append(mean-std_err)
  transfer_intervals_higher.append(mean+std_err)
  transfer_means.append(mean)
# %%
'''
x_values = [1,2,3,5,7,10,100]

# Plotting threshold
plt.figure(figsize=(10,6), dpi=400)
plt.plot(x_values, threshold_means, 'o-', label='Threshold', color='grey')
plt.fill_between(x_values, threshold_intervals_lower, threshold_intervals_higher, alpha=0.3, color='grey')

# Plotting random
plt.plot(x_values, random_means, 'o-', label='Random', color='red')
plt.fill_between(x_values, random_intervals_lower, random_intervals_higher, alpha=0.3, color='red')

# Plotting 3D-RPL
plt.plot(x_values, rpl_means, 'o-', label='3D-RPL', color = 'green')
plt.fill_between(x_values, rpl_intervals_lower, rpl_intervals_higher, alpha=0.3, color='green')

# Plot transfer
plt.plot(x_values, transfer_means, 'o-', label='Transfer', color = 'blue')
plt.fill_between(x_values, transfer_intervals_lower, transfer_intervals_higher, alpha=0.3, color='blue')


plt.xlabel('Percentage of labeled training data utilized in training\n(Percentage of all labeled data utilized for training)')
plt.ylabel('Average Test Dice Score')
plt.xscale('log')
plt.xticks(x_values, ['1%\n(0.8%)', '2%\n(1.6%)', '3%\n(2.4%)', '5%\n(4%)', '7%\n(5.6%)', '10%\n(8%)', '100%\n(80%)'])  # Set custom x-ticks

#plt.yticks(range(0.35, 0.6))  # Set y-ticks from 0% to 60%
plt.ylim(0.05, 0.50)  # Set y-axis range from 0% to 60%
#plt.yscale('log')
plt.legend()
plt.title('Test Dice Scores vs. Percentage Labeled Training Data')
plt.show()


####### ZOOMED IN PLOT RANGE 1% - 10% ############
# Plotting random
plt.figure(figsize=(10,6), dpi=400)
fig, ax = plt.subplots(figsize=(10,6), dpi=400)

# Plot threshold
plt.plot(x_values[:-1], threshold_means[:-1], 'o-', label='Threshold', color='grey')
plt.fill_between(x_values[:-1], threshold_intervals_lower[:-1], threshold_intervals_higher[:-1], alpha=0.3, color='grey')

# Plot random
plt.plot(x_values[:-1], random_means[:-1], 'o-', label='Random', color='red')
plt.fill_between(x_values[:-1], random_intervals_lower[:-1], random_intervals_higher[:-1], alpha=0.3, color='red')

# Plotting 3D-RPL
plt.plot(x_values[:-1], rpl_means[:-1], 'o-', label='3D-RPL', color = 'green')
plt.fill_between(x_values[:-1], rpl_intervals_lower[:-1], rpl_intervals_higher[:-1], alpha=0.3, color='green')

# Plot transfer
plt.plot(x_values[:-1], transfer_means[:-1], 'o-', label='Transfer', color = 'blue')
plt.fill_between(x_values[:-1], transfer_intervals_lower[:-1], transfer_intervals_higher[:-1], alpha=0.3, color='blue')


plt.xlabel('Percentage of labeled training data utilized in training\n(Percentage of all labeled data utilized for training)')
plt.ylabel('Average Test Dice Score')
labels = ['1%\n(0.8%)', '2%\n(1.6%)', '3%\n(2.4%)', '5%\n(4%)', '7%\n(5.6%)', '10%\n(8%)']

#plt.yticks(range(0.35, 0.6))  # Set y-ticks from 0% to 60%
plt.ylim(0.05, 0.50)  # Set y-axis range from 0% to 60%
plt.legend()
plt.title('Test Dice Scores vs. Percentage Labeled Training Data - Zoomed In')
plt.show()



plt.xlabel('Percentage of labeled training data utilized in training\n(Percentage of all labeled data utilized for training)')
plt.ylabel('Average Test Dice Score')
labels = ['1%\n(0.8%)', '2%\n(1.6%)', '3%\n(2.4%)', '5%\n(4%)', '7%\n(5.6%)', '10%\n(8%)']

# Create a second x-axis
plt.twiny()

# Set the position and labels for the second x-axis
plt.xlim(plt.xlim())
plt.xticks(x_values[:-1], labels)

plt.ylim(0.05, 0.50)  # Set y-axis range from 0.05 to 0.50
plt.legend()
plt.title('Test Dice Scores vs. Percentage Labeled Training Data - Zoomed In')
plt.show()
'''
# %%
import matplotlib.pyplot as plt
import numpy as np

x_values = [1,2,3,5,7,10,100]


# Plotting random
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=400)
ax2 = ax1.twiny()  # Create a second x-axis

# Plot threshold
ax1.plot(x_values, threshold_means, 'o-', label='Threshold', color='grey')
ax1.fill_between(x_values, threshold_intervals_lower, threshold_intervals_higher, alpha=0.3, color='grey')

# Plot random
ax1.plot(x_values, random_means, 'o-', label='Random', color='red')
ax1.fill_between(x_values, random_intervals_lower, random_intervals_higher, alpha=0.3, color='red')

# Plotting 3D-RPL
ax1.plot(x_values, rpl_means, 'o-', label='3D-RPL', color='green')
ax1.fill_between(x_values, rpl_intervals_lower, rpl_intervals_higher, alpha=0.3, color='green')

# Plot transfer
ax1.plot(x_values, transfer_means, 'o-', label='Transfer', color='blue')
ax1.fill_between(x_values, transfer_intervals_lower, transfer_intervals_higher, alpha=0.3, color='blue')

ax1.set_xlabel('Percentage of labeled training data')
ax1.set_ylabel('Average Test Dice Score')

# Set the labels for the first x-axis
labels1 = ['1', '2%', '3%', '5%', '7%', '10%', '100%']
ax1.set_xscale('log')
ax1.set_xticks(x_values)
ax1.set_xticklabels(labels1)



# Set the labels for the second x-axis
labels2 = ['0.8%', '1.6%', '2.4%', '4%', '5.6%', '8%', '80%']
ax2.set_xscale('log')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(x_values)
ax2.set_xticklabels(labels2, fontstyle='italic')  # Set fontstyle to italic
ax2.set_xlabel('Percentage of entire dataset', fontstyle = "italic")



ax1.set_ylim(0.05, 0.50)  # Set y-axis range from 0.05 to 0.50
ax1.set_title('Test Dice Scores vs. Percentage Labeled Data', pad = 10, fontsize = 16)

plt.show()







# Plotting random
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=400)
ax2 = ax1.twiny()  # Create a second x-axis

# Plot threshold
ax1.plot(x_values[:-1], threshold_means[:-1], 'o-', label='Threshold', color='grey')
ax1.fill_between(x_values[:-1], threshold_intervals_lower[:-1], threshold_intervals_higher[:-1], alpha=0.3, color='grey')

# Plot random
ax1.plot(x_values[:-1], random_means[:-1], 'o-', label='Random', color='red')
ax1.fill_between(x_values[:-1], random_intervals_lower[:-1], random_intervals_higher[:-1], alpha=0.3, color='red')

# Plotting 3D-RPL
ax1.plot(x_values[:-1], rpl_means[:-1], 'o-', label='3D-RPL', color='green')
ax1.fill_between(x_values[:-1], rpl_intervals_lower[:-1], rpl_intervals_higher[:-1], alpha=0.3, color='green')

# Plot transfer
ax1.plot(x_values[:-1], transfer_means[:-1], 'o-', label='Transfer', color='blue')
ax1.fill_between(x_values[:-1], transfer_intervals_lower[:-1], transfer_intervals_higher[:-1], alpha=0.3, color='blue')

ax1.set_xlabel('Percentage of labeled training data')
ax1.set_ylabel('Average Test Dice Score')

# Set the labels for the first x-axis
labels1 = ['1', '2%', '3%', '5%', '7%', '10%']
ax1.set_xticks(x_values[:-1])
ax1.set_xticklabels(labels1)

# Set the labels for the second x-axis
labels2 = ['0.8%', '1.6%', '2.4%', '4%', '5.6%', '8%']
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(x_values[:-1])
ax2.set_xticklabels(labels2, fontstyle='italic')  # Set fontstyle to italic
ax2.set_xlabel('Percentage of entire dataset', fontstyle = "italic")


ax1.set_ylim(0.05, 0.50)  # Set y-axis range from 0.05 to 0.50
ax1.set_title('Test Dice Scores vs. Percentage Labeled Data', pad = 10, fontsize = 16)

plt.show()








# %%
