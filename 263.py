import math

l1 = 83.55
l2 = 73.29

def theta(x, y):
    numerator = x * x + pow(94.13 - y, 2) - l1 * l1 - l2 * l2
    denominator = -2 * l1 * l2

    theta = 180 - math.asin(l2*math.sin(math.acos(numerator/denominator)) / math.sqrt(l1 * l1 + l2 * l2 - 2*l1*l2 * (numerator/denominator))) - math.atan(x / (94.13 - y))
    return theta

print(theta(0, 0))