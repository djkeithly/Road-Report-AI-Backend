Procure Data for csv_editing.py from: https://cris.dot.state.tx.us/public/Query/app/query-builder/disclaimer

#   #   #   #   #   #   #   #   #   #   #   #
#               Steps to run                #
#   #   #   #   #   #   #   #   #   #   #   #

(1) Download the crash data from the above link, place in csv directory, and set rawCSV to whatever the file is named
(2) Ensure that the variables in CreateTrainingData.py are correct. Default values are in Variable List
(3) Run:

py csv/CreateTrainingData.py

(4) Remember to be patient, while there is logging to notify you of what is happening, notifications can be infrequent. 
    So long as no error message appears or abortion happens, the files should be runing
(5) If the center of roads roads are needed, add --road to the end of the command in step 3:

py csv/CreateTrainingData.py --road

#   #   #   #   #   #   #   #   #   #   #   #
#               Variable List               #
#   #   #   #   #   #   #   #   #   #   #   #

Main file to edit to customize what data is generated is CreateTrainingData.py. No other file should need to be edited

# state = "TX"
    The State to be looked into
# ctry = "US"
    Country. Should always be united states
# cities = True
    If weather data should get all the cities in the counties (True) or just counties (False)
    Should be set to false for reasonable data sizes

## Right Now This Only Works with 1 year. No pipeline for multiple years exist yet
# years = [ 2025]
    What years the weather should be fetching. Should match up to the years of Crash Data you have provided
    

# rawCSV = "raw_crash_data.csv"
    The name of the crash data csv you downloaded and placed in the csv directory. Should be set to the name of the file you downloaded