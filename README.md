# Understanding-the-spreading-patterns-of-connected-vehicle-viruses
## Data
In this study, we use four data collected by the United States Census Bureau, which are all available online. The used GIS data of US counties is available at https://www2.census.gov/geo/tiger/TIGER2015/COUNTY/. The number of commuters leaving home to go to work in each time period and the number of commuters in each trip duration range are obtained from https://data.census.gov/cedsci/table?q=Commuting&tid=ACSDT5Y2015.B08011 and https://data.census.gov/cedsci/table?q=Commuting&tid=ACSDT5Y2015.B08012. The used travel survey data, 2011-2015 5-Year ASC Commuting Flows, is available at  https://www.census.gov/data/tables/2015/demo/metro-micro/commuting-flows-2015.html. These data are all stored in the ‘Files’ folder.

## Model
Our model is implemented through five modules, all of which are coded by Python:<br>
1.Data_process.py preprocesses the used data.<br>
2.Movement.py simulates the movements of connected vehicles.<br>
3.Network.py generates the vehicle communication network.<br>
4.WiFi.py simulates the spread of WiFi viruses.<br>
5.GSM.py simulates the spread of GSM viruses.<br>

## Environment
All of the programs are run and tested on Python 3.8.3.<br>

## How to run
Please run the programs in the following order.<br>
1.Data_process.py<br>
2.Movement.py<br>
3.Network.py<br>
4.WiFi.py and GSM.py<br>
Please determine the model parameters before running the programs. Some intermediate results need to be generated using ArcGIS. The overall running time of the programs on three workstations is 15 days (each workstation is with a 3.70 GHz Intel Core CPU and 64 GB memory).
