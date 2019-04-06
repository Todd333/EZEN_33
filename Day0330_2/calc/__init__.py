from calc.Calc import Calc

def main():
    calc = Calc(3,7)
    print("{} +{} ={}".format(calc.first,calc.second,calc.sum()))

if __name__ == '__main__':
    main()