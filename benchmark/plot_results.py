import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
print(df)

plt.figure()
plt.bar(df['program'], df['time'])
plt.ylabel('Time (seconds)')
plt.title('GPU vs CPU MD5 Brute-Force Time')
plt.tight_layout()
plt.show()
