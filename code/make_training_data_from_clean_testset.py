import csv
import random

input_file = 'TESTclean.csv'
output_file = 'TRAIN_CLEAN.csv'
num_samples = 50000

data = []

with open(input_file, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        data.append(row)

random.shuffle(data)
selected_samples = data[:num_samples]

with open(output_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)
    csv_writer.writerows(selected_samples)
    