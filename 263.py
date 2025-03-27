import numpy as np

l1 = 83.55
l2 = 180
l3 = 73.29
x = 212.04
y = 37.82

i = 166
j = 83.55

def thetaprime(x, y):
    numerator = x * x + pow(94.13 - y, 2) - l1 * l1 - l2 * l2
    denominator = -2 * l1 * l2
    thetaprime = np.degrees(np.arccos(numerator/denominator))
    return thetaprime


def theta(x, y):
    theta2 = np.degrees(np.arcsin((l2*(np.sin(np.radians(thetaprime(x, y))))) / (np.sqrt(l1*l1 + l2*l2 - 2*l1*l2*(np.cos(np.radians(thetaprime(x, y))))))))
    theta3 = np.degrees(np.arctan(x / (94.13 - y)))
    # print(thetaprime(x, y))
    # print(theta2)
    # print(theta3)
    return (180 - theta2 - theta3)



def phi(x, y):
    e = l1*np.sin(np.radians(theta(x, y)))
    f = l3*np.cos(np.radians(90 + theta(x, y) - thetaprime(x, y)))
    x2 = e + f

    d = l1*np.cos(np.radians(theta(x, y)))
    g = l3*np.sin(np.radians(90 + theta(x, y) - thetaprime(x, y)))
    y2 = 94.13 + d - g
    def phi1():
        answer = np.degrees(np.arctan(x2 / y2))
        return answer
    
    def phi2():
        h = np.sqrt(x2*x2 + y2*y2)
        answer = np.degrees(np.arccos((i*i - h*h - j*j) / (-2*h*j)))
        return answer
    
    return phi1() + phi2()

x = float(input("Enter x: "))
y = float(input("Enter y: "))

print("Theta = " + str(theta(x, y)) + " degrees")
print("Phi = " + str(phi(x, y)) + " degrees")


