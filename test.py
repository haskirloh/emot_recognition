import matplotlib.pyplot as plt

categories = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
values = [1, 2, 3, 4, 5, 6, 7]

fig1, ax1 = plt.subplots()
ax1.pie(values, labels=categories)
plt.savefig('diagram.png')
