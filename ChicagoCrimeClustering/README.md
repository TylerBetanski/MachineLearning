# Chicago Crime Clustering

Goal: Use K-means clustering to investigate various crimes in Chicago

Each datapoint is a coordinate (Long, Lat) alongside the type of crime committed. The data was cleaned to only include crimes committed within the Chicago Metropolitan area (-88 to -87.5 LONG, 41.6 to 42 LAT).

The various crimes were combined into 5 categories:
><b>Larceny</b>: Robbery, Burglary, Theft\
><b>Violent Crimes</b>: Battery, Assault, Homicide, Intimidation\
><b>Drug Crimes</b>: Narcotics & Other Narcotics Violations\
><b>Sex Crimes</b>: Stalking, Crim Sexual Assault, Sex Offense, Kidnapping, Obscenity, Human Trafficking\
><b>Weapons Violations</b>: Weapons Violation or Concealed Carry Violation

Initially, the crimes are clustered based on their type, seperating the city into 8 clusters for each crime type. A heatmap
is used to show the distribution of crimes.

## Results:
<img src="https://github.com/user-attachments/assets/08cbe440-31eb-4f73-a05f-be40bf5d7894" width="300"></img>
<img src="https://github.com/user-attachments/assets/4cb4d79f-d3d2-43ae-b554-bdc2b1c53c05" width="300"></img>
<img src="https://github.com/user-attachments/assets/10c3efd6-e420-4ad3-b20f-e9198742bb99" width="300"></img>
<img src="https://github.com/user-attachments/assets/af65324b-0687-49bb-8f8d-e87e7ca5d4c3" width="300"></img>
<img src="https://github.com/user-attachments/assets/daa844c8-85e7-4189-9416-66d3ea3a3856" width="300"></img>

Next the city was clustered into 22 groups using all the crimes to compare the real police precincts with calculated ones.\
<img src="https://github.com/user-attachments/assets/7fa18996-76c4-46c6-9e6a-099f48f6e473" width="350"></img>
<img src="https://github.com/user-attachments/assets/6e809b72-69c8-406d-a98b-997e86725073" width="300"></img>

The clustering does an alright job of dividing the city. The areas near the coast have similar numbers of precincts,
while the north and south have fewer and larger precincts.


## References:
Dataset sourced from: https://www.kaggle.com/datasets/currie32/crimes-in-chicago/data  
The Chicago map used in the heatmaps was created with: https://maps.co/gis/  
Chicago Precinct map sourced from: https://chicagopd.maps.arcgis.com/apps/instant/nearby/index.html?appid=11a23d43d62b4f929dd0ec0f8c013506
