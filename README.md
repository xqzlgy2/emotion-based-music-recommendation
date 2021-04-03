# emotion-based-music-recommendation
Recommend Spotify songs based on user's playlist and emotion state.



## File name and functions:

### jiaqideng branch

#### CSV files
* 1 processed_data_full.csv: Whole data Qinzhou precessed in the dataset.
* 2 Front_Left.csv: One playlist in the Spotify, it lists all the 100 songs in Front_Left playlist. 
* 3 attribute.csv: It list 18 attributes about these 100 songs in the Front_Left playlist. 

#### Python files
* 1 Classifiers.py: Try different sklearn classify method and see which one has the highest accurancy (Random forest: about 60% ).
* 2 Spotify_API.py: Get playlist's information in Spotify and write them to the csv file

