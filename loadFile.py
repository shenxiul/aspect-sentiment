import csv


def load(filename):
    input_file = open(filename, 'r')
    csv_reader = csv.reader(input_file)
    data = []
    for line in csv_reader:
        data.append({'review': line[0].lower(), 'aspect': line[1], 'rating': float(line[2])})
    return data

if __name__ == '__main__':
    data_set = load('./data/final_review_set.csv')
