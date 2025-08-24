import statistics
import preprocessing
import sys

state = 'Select Param'
param = statistics.meanfromyears
strparam = ''
countrycode = ''
codedict = preprocessing.codetoname(sys.argv[1])
fromyear = 0
toyear = 0
years = preprocessing.years(sys.argv[1])
while True:
    while state == 'Select Param':
        print('Welcome! Select one of the following statistical parameters:\n1. Mean population\n2. Median population\n3. Mode of population\n4. Standard deviation of population\n5. Variance of population\n6. Percentage change in population\n7. Exit ')
        try:    
            choice = int(input('User selects -> '))
            if choice >= 1 and choice <= 7:
                if choice == 1:
                    param = statistics.meanfromyears
                    strparam = 'mean in population'
                elif choice == 2:
                    param = statistics.medianfromyears
                    strparam = 'median in population'
                elif choice == 3:
                    param = statistics.modefromyears
                    strparam = 'mode in population'
                elif choice == 4:
                    param = statistics.stddevfromyears
                    strparam = 'standard deviation in population'
                elif choice == 5:
                    param = statistics.variancefromyears
                    strparam = 'variance in population'
                elif choice == 6:
                    param = statistics.percentchangefromyears
                    strparam = 'change in population in percent'
                elif choice == 7:
                    quit()
                state = 'Select Country'
            else:
                print('Invalid input!')
        except ValueError:
            print('Invalid input!')
        
    while state == 'Select Country':
        print('Select country of interest:')
        for i in range(len(list(codedict.keys()))):
            print(str(i+1) + '. '+list(codedict.keys())[i])
        try:
            choice = int(input('User selects -> '))
            if choice >= 1 and choice <= len(list(codedict.keys())):
                countrycode = list(codedict.keys())[choice-1]
                state = 'Select From'
            else:
                print('Invalid input!')
        except ValueError:
            print('Invalid input!')
        
    while state == 'Select From':
        print('Select from year:')
        for i in range(len(years)):
            print(str(i+1) + '.', years[i])
        try:
            choice = int(input('User selects -> '))
            if choice >= 1 and choice <= len(years):
                fromyear = years[choice-1]
                state = 'Select To'
            else:
                print('Invalid input!')
        except ValueError:
            print('Invalid input!')
        
    while state == 'Select To':
        print('Select to year:')
        for i in range(len(years)):
            print(str(i+1) + '.', years[i])
        try:
            choice = int(input('User selects -> '))
            if choice >= 1 and choice <= len(years):
                toyear = years[choice-1]
                state = 'Output'
            else:
                print('Invalid input!')
        except ValueError:
            print('Invalid input!')
    if state == 'Output':
        print('The', strparam, 'of country', codedict[countrycode], 'from year', fromyear, 'to', toyear, 'is', param(preprocessing.preprocess(sys.argv[1]), countrycode, int(fromyear), int(toyear)))
        state = 'Select Param'